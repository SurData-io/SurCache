"""Six worker threads. They only hand each other a queue item or a wake-up.

lookup, preload, H2D, D2H, store, and storeGC never call a function that
waits for a different stream to finish. A lookup or a SurData read may
block inside its own thread. The compute hooks only record events and
enqueue one snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import mmap
import os
import struct
import sys
import threading
import time

_LOG = logging.getLogger(__name__)

from surdata_vllm.config import load_direct_phase_diag_ms, load_external_kv_worker_count
from surdata_vllm.dram_queue import (
    STATUS_H2D_IN_FLIGHT,
    STATUS_INIT,
    STATUS_L1_READING,
    STATUS_LOOKUP_PENDING,
    STATUS_PRELOADED,
    DramPipelineQueue,
)
from surdata_vllm.kvb import KvGeometry

PHASE_NONE = 0
PHASE_PENDING = 1
PHASE_MISS = 2
# Lookup retries while another live request may still publish this prefix.
# The retry runs on the lookup thread and does not take a GPU step. After
# this many seconds the current 0 stands, so a slow store cannot hold a
# lookup-window slot until that save finally returns.
PUBLISH_WAIT_S = 2.0
PHASE_INIT = 3
PHASE_READING = 4
PHASE_PRELOADED = 5
PHASE_H2D = 6
PHASE_HBM = 7
PHASE_CLOSED = 8


class CopyKind(enum.IntEnum):
    """One device copy in ``_jobs``. Values stay clear of phases and queue statuses."""

    H2D = 32
    D2H = 33
    PARK = 34

_MAGIC = 0x53564431
_SLOTS = 1024
_SLOT = 128
_HEAD = 16
_BOOT_ID = 0


def _process_boot_id() -> int:
    """One id per process start. A reused shm file from an older start is wiped."""
    global _BOOT_ID
    if _BOOT_ID == 0:
        import secrets

        value = secrets.randbits(32)
        _BOOT_ID = value if value != 0 else 1
    return _BOOT_ID


def matched_from_phase(phase: int, hit_tokens: int, computed: int):
    """Only PRELOADED may allocate HBM. H2D is already in flight."""
    if phase in (PHASE_NONE, PHASE_PENDING, PHASE_INIT, PHASE_READING, PHASE_H2D):
        return None, False
    if phase in (PHASE_MISS, PHASE_CLOSED, PHASE_HBM):
        return 0, False
    if phase == PHASE_PRELOADED:
        count = max(0, int(hit_tokens) - int(computed))
        if count <= 0:
            return 0, False
        return count, True
    return None, False


def status_board_path(pid: int) -> str:
    return "/dev/shm/surdata_vllm_kv_status_%d" % int(pid)


def _read_board_header(path: str):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = os.read(fd, _HEAD)
    finally:
        os.close(fd)
    if len(raw) < 4:
        return None
    magic = struct.unpack_from("<I", raw, 0)[0]
    slots = struct.unpack_from("<I", raw, 4)[0] if len(raw) >= 8 else 0
    boot = struct.unpack_from("<I", raw, 8)[0] if len(raw) >= 12 else 0
    return magic, slots, boot


def _open_own_board(path: str) -> "StatusBoard":
    """Open this pid's board. A magic file from another boot is rewritten."""
    header = _read_board_header(path)
    if header is not None and header[0] == _MAGIC and int(header[2]) == _process_boot_id():
        return StatusBoard(path, create=False)
    if header is not None and header[0] == _MAGIC:
        return StatusBoard(path, create=True, reset=True)
    return StatusBoard(path, create=True)


def open_scheduler_board() -> "StatusBoard":
    """Open this process's board. Create it when the worker has not already."""
    return _open_own_board(status_board_path(os.getpid()))


def open_worker_board(wait_s: float = 2.0) -> "StatusBoard":
    """Open the EngineCore board.

    Same process uses this pid. A worker whose own file is missing opens the
    parent pid read-only when that file already has the magic, and does not
    truncate it when the boot id differs. v0.33 constructs the worker connector
    before the scheduler in that same process, so if neither file exists the
    worker creates this pid's board.
    """
    deadline = 0.0 if wait_s <= 0 else _mono() + float(wait_s)
    while True:
        own = status_board_path(os.getpid())
        header = _read_board_header(own)
        if header is not None and header[0] == _MAGIC:
            return _open_own_board(own)
        parent_pid = os.getppid()
        parent = status_board_path(parent_pid)
        parent_header = None if parent_pid == os.getpid() else _read_board_header(parent)
        if parent_header is not None and parent_header[0] == _MAGIC:
            return StatusBoard(parent, create=False, readonly=True)
        if _mono() >= deadline:
            break
        _sleep(0.02)
    return StatusBoard(status_board_path(os.getpid()), create=True)


def _mono() -> float:
    import time

    return time.monotonic()


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_MU = threading.Lock()


def _publish_lock(path: str) -> threading.Lock:
    """One lock per shm file. Two StatusBoard objects share a file but not self._lock."""
    with _PATH_LOCKS_MU:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[path] = lock
        return lock


@dataclass
class StatusView:
    generation: int = 0
    phase: int = PHASE_NONE
    hit_tokens: int = 0
    closed: int = 0


class StatusBoard:
    """Seqlock table indexed by a scheduler slot. A reader retries a torn slot."""

    def __init__(self, path: str | None = None, create: bool = False, reset: bool = False, readonly: bool = False):
        self._path = path
        self._mm = None
        self._fd = None
        self._readonly = bool(readonly)
        self._at: list[StatusView | None] = [None] * _SLOTS
        self._lock = threading.Lock()
        if not path:
            return
        size = _HEAD + _SLOTS * _SLOT
        if self._readonly:
            self._fd = os.open(path, os.O_RDONLY)
            self._mm = mmap.mmap(self._fd, size, prot=mmap.PROT_READ)
            return
        flags = os.O_RDWR | (os.O_CREAT if create or reset else 0)
        self._fd = os.open(path, flags, 0o600)
        rewrite = False
        if create or reset:
            raw = os.read(self._fd, _HEAD)
            os.lseek(self._fd, 0, os.SEEK_SET)
            magic_ok = len(raw) >= 4 and struct.unpack_from("<I", raw, 0)[0] == _MAGIC
            if reset or not magic_ok:
                rewrite = True
                os.ftruncate(self._fd, 0)
                os.ftruncate(self._fd, size)
        self._mm = mmap.mmap(self._fd, size)
        if rewrite:
            self._mm.seek(0)
            self._mm.write(struct.pack("<IIII", _MAGIC, _SLOTS, _process_boot_id(), 0))

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def publish_at(self, slot: int, generation: int, phase: int, hit_tokens: int, closed: int = 0) -> None:
        slot = int(slot)
        generation = int(generation) & 0xFFFFFFFF
        phase = int(phase)
        hit_tokens = int(hit_tokens)
        closed = int(closed)
        if slot < 0 or slot >= _SLOTS or self._readonly:
            return
        if self._mm is None:
            with self._lock:
                self._at[slot] = StatusView(generation, phase, hit_tokens, closed)
            return
        with _publish_lock(self._path):
            off = _HEAD + slot * _SLOT
            seq = struct.unpack_from("<I", self._mm, off)[0]
            struct.pack_into("<I", self._mm, off, seq + 1)
            struct.pack_into("<IIII", self._mm, off + 4, generation, phase, hit_tokens, closed)
            struct.pack_into("<I", self._mm, off, seq + 2)

    def get_at(self, slot: int) -> StatusView | None:
        slot = int(slot)
        if slot < 0 or slot >= _SLOTS:
            return None
        if self._mm is None:
            with self._lock:
                view = self._at[slot]
                if view is None or int(view.generation) == 0:
                    return None
                return StatusView(view.generation, view.phase, view.hit_tokens, view.closed)
        for _ in range(8):
            off = _HEAD + slot * _SLOT
            seq1 = struct.unpack_from("<I", self._mm, off)[0]
            if seq1 & 1:
                continue
            generation, phase, hit, closed = struct.unpack_from("<IIII", self._mm, off + 4)
            seq2 = struct.unpack_from("<I", self._mm, off)[0]
            if seq1 != seq2:
                continue
            if int(generation) == 0:
                return None
            return StatusView(int(generation), int(phase), int(hit), int(closed))
        return None

    def release(self, slot: int, generation: int) -> None:
        """Zero this slot only while it still holds this generation."""
        slot = int(slot)
        generation = int(generation) & 0xFFFFFFFF
        if slot < 0 or slot >= _SLOTS or generation == 0 or self._readonly:
            return
        if self._mm is None:
            with self._lock:
                current = self._at[slot]
                if current is None or int(current.generation) != generation:
                    return
                self._at[slot] = None
            return
        with _publish_lock(self._path):
            off = _HEAD + slot * _SLOT
            existing_gen = struct.unpack_from("<I", self._mm, off + 4)[0]
            if int(existing_gen) != generation:
                return
            seq = struct.unpack_from("<I", self._mm, off)[0]
            struct.pack_into("<I", self._mm, off, seq + 1)
            struct.pack_into("<IIII", self._mm, off + 4, 0, 0, 0, 0)
            struct.pack_into("<I", self._mm, off, seq + 2)


@dataclass
class ConnectorMeta:
    lookups: list = field(default_factory=list)
    allocs: list = field(default_factory=list)
    finished: list = field(default_factory=list)
    preempted: list = field(default_factory=list)
    save_hints: list = field(default_factory=list)
    next_bytes: int = 0


class SchedulerFront:
    """Hands out board slots. Reads the board. Does not call the SDK."""

    def __init__(self, board: StatusBoard):
        self.board = board
        self._pending: dict[str, dict] = {}
        self._sent: dict[str, int] = {}
        self._free = list(range(_SLOTS - 1, -1, -1))
        self._slot_gen = [0] * _SLOTS
        self._allocs = []
        self._finished = []
        self._preempted = []
        self._hints = []
        self._blocks: dict[str, list] = {}
        self._answered_preloaded: set[str] = set()

    def _fields(self, req_id: str):
        item = self._pending.get(str(req_id))
        if item is None:
            return 0, -1
        return int(item["generation"]), int(item["slot"])

    def on_new_request(self, req_id: str, tokens: list) -> int:
        key = str(req_id)
        tokens = [int(tok) for tok in tokens]
        current = self._pending.get(key)
        if current is not None:
            slot = int(current["slot"])
            generation = int(current["generation"])
            current["tokens"] = tokens
            self._sent.pop(key, None)
            self.board.publish_at(slot, generation, PHASE_PENDING, 0, 0)
            return generation
        if not self._free:
            _LOG.warning("status board free list is empty; request %s was not published", key)
            return 0
        slot = self._free.pop()
        generation = int(self._slot_gen[slot]) + 1
        self._slot_gen[slot] = generation
        self._pending[key] = {"tokens": tokens, "generation": generation, "slot": slot}
        self.board.publish_at(slot, generation, PHASE_PENDING, 0, 0)
        return generation

    def note_alloc(self, req_id: str, block_ids: list, external_tokens: int) -> None:
        generation, slot = self._fields(req_id)
        self._allocs.append(
            {
                "req_id": str(req_id),
                "generation": generation,
                "slot": slot,
                "block_ids": [int(block) for block in block_ids],
                "external_tokens": int(external_tokens),
            }
        )

    def note_finished(self, req_id: str, block_ids: list, num_tokens: int) -> None:
        key = str(req_id)
        generation, slot = self._fields(key)
        self._finished.append(
            {
                "req_id": key,
                "generation": generation,
                "slot": slot,
                "block_ids": [int(block) for block in block_ids],
                "num_tokens": int(num_tokens),
            }
        )
        self._blocks.pop(key, None)
        if self._pending.pop(key, None) is None or slot < 0:
            return
        self.board.release(slot, generation)
        self._free.append(slot)

    def remember_blocks(self, req_id: str, block_ids: list, num_tokens: int, replace: bool) -> None:
        """Keep this request's full block table and publish one save hint.

        ``replace`` installs ``block_ids`` as the whole table. Otherwise the new
        ids are appended. An empty update keeps the table already stored.
        ``num_tokens`` is how many tokens the forward that is about to run will
        have written when it returns. A non-positive count is not published.
        """
        key = str(req_id)
        incoming = [int(block) for block in (block_ids or [])]
        current = list(self._blocks.get(key, []))
        if replace and incoming:
            current = incoming
        elif incoming:
            seen = set(current)
            for block in incoming:
                if block not in seen:
                    current.append(block)
                    seen.add(block)
        if current:
            self._blocks[key] = current
        num = int(num_tokens)
        if num <= 0 or not current:
            return
        self.note_save(key, current, num)

    def note_save(self, req_id: str, block_ids: list, num_tokens: int) -> None:
        key = str(req_id)
        ids = [int(block) for block in block_ids]
        if ids:
            self._blocks[key] = ids
        generation, slot = self._fields(key)
        self._hints.append(
            {
                "req_id": str(req_id),
                "generation": generation,
                "slot": slot,
                "block_ids": [int(block) for block in block_ids],
                "num_tokens": int(num_tokens),
            }
        )

    def note_preempt(self, req_id: str) -> None:
        key = str(req_id)
        self._blocks.pop(key, None)
        item = self._pending.get(key)
        if item is None:
            self._preempted.append({"req_id": key, "generation": 0, "slot": -1})
            return
        generation = int(item["generation"])
        slot = int(item["slot"])
        self._preempted.append({"req_id": key, "generation": generation, "slot": slot})

    def build_meta(self) -> ConnectorMeta:
        meta = ConnectorMeta()
        for req_id, item in self._pending.items():
            generation = int(item["generation"])
            if self._sent.get(req_id) == generation:
                continue
            self._sent[req_id] = generation
            meta.lookups.append(
                {
                    "req_id": req_id,
                    "generation": generation,
                    "slot": int(item["slot"]),
                    "tokens": list(item["tokens"]),
                }
            )
        meta.allocs = self._allocs
        meta.finished = self._finished
        meta.preempted = self._preempted
        meta.save_hints = self._hints
        self._allocs = []
        self._finished = []
        self._preempted = []
        self._hints = []
        self._answered_preloaded.clear()
        return meta

    def _unanswered_preloaded(self, skip_req: str) -> bool:
        """True when some other request is PRELOADED and this step has not answered it."""
        for req_id, item in self._pending.items():
            if req_id == skip_req or req_id in self._answered_preloaded:
                continue
            view = self.board.get_at(int(item["slot"]))
            if view is None or int(view.generation) != int(item["generation"]):
                continue
            if int(view.phase) == PHASE_PRELOADED:
                return True
        return False

    def matched_tokens(self, req_id: str, computed: int):
        """One request. PRELOADED hits go first. A miss is only for a request
        asked after every PRELOADED request this step has already been answered.
        The scheduler stops asking when an allocation fails, so a later miss
        means HBM still had room. The board phase is not rewritten here.
        """
        key = str(req_id)
        local = self._pending.get(key)
        if local is None:
            return None, False
        view = self.board.get_at(int(local["slot"]))
        if view is None or int(view.generation) != int(local["generation"]):
            return None, False
        phase = int(view.phase)
        if phase == PHASE_PRELOADED:
            self._answered_preloaded.add(key)
            return matched_from_phase(phase, view.hit_tokens, computed)
        if phase == PHASE_H2D:
            return None, False
        if phase in (PHASE_MISS, PHASE_CLOSED, PHASE_HBM):
            return matched_from_phase(phase, view.hit_tokens, computed)
        if self._unanswered_preloaded(key):
            return None, False
        if phase in (PHASE_NONE, PHASE_PENDING, PHASE_INIT, PHASE_READING):
            return 0, False
        return matched_from_phase(phase, view.hit_tokens, computed)


@dataclass
class _Block:
    block_id: int
    valid_tokens: int
    src: dict
    events: dict
    saved: bool = False
    inflight: bool = False
    d2h_done: bool = False
    token_index: int = 0
    stride: int = 0
    src_epoch: int = 0
    # One event recorded at the end of the forward. per_layer leaves this empty.
    save_event: object = None


@dataclass
class _Req:
    req_id: str
    generation: int
    tokens: list
    phase: int = PHASE_PENDING
    hit_tokens: int = 0
    finished: bool = False
    num_tokens: int = 0
    block_ids: list = field(default_factory=list)
    external_tokens: int = 0
    # True only after ingest_blocks installs the scheduler's full block table.
    # Alloc lists can start after a local prefix, so H2D must not slice those.
    full_block_table: bool = False
    blocks: dict = field(default_factory=dict)
    layer_base: dict = field(default_factory=dict)
    layer_events: dict = field(default_factory=dict)
    closed: bool = False
    slot: int = -1
    sending_published: bool = False
    open_blocks: list = field(default_factory=list)
    built_tokens: int = 0
    base_epoch: int = 0


def _producer_recorded(block: _Block) -> bool:
    if block.save_event is not None:
        return True
    if not block.events:
        return False
    return all(ev is not None for ev in block.events.values())


_TRACE_ON = ("1", "true", "yes", "on")
_trace_suffix = None
_trace_emit_patched = False


def _decode_trace_enabled() -> bool:
    return os.environ.get("SURDATA_DECODE_TRACE", "0").strip().lower() in _TRACE_ON


def _attach_step_suffix(suffix: str) -> bool:
    """Hang scan fields on the next ``SURDATA_DECODE_TRACE step=`` line.

    decode_trace emits that line after wait_for_save returns. Patching its
    emitter keeps the counters on the existing line. A missing module prints
    one fallback line instead of a per-layer file.
    """
    global _trace_suffix, _trace_emit_patched
    mod = sys.modules.get("surdata_lmcache.decode_trace")
    emit = getattr(mod, "_emit", None) if mod is not None else None
    if not callable(emit):
        return False
    _trace_suffix = suffix
    if _trace_emit_patched:
        return True
    orig = emit

    def _emit_with_scan(msg: str) -> None:
        global _trace_suffix
        extra = _trace_suffix
        if extra and isinstance(msg, str) and msg.startswith("SURDATA_DECODE_TRACE step="):
            msg = msg + " " + extra
            _trace_suffix = None
        orig(msg)

    mod._emit = _emit_with_scan
    _trace_emit_patched = True
    return True


class TransferResults:
    def __init__(self):
        self.finished_sending = set()
        self.finished_recving = set()
        self.failed_recving = set()


class WorkerRuntime:
    def __init__(
        self,
        sdk,
        pool=None,
        native=None,
        geometry: KvGeometry | None = None,
        policy: str = "finish_drop",
        save_mode: str = "per_layer",
        board: StatusBoard | None = None,
        preload_pool=None,
        store_pool=None,
        preload_bytes: int | None = None,
        store_bytes: int | None = None,
        lookup_limit: int = 32,
        publish_wait_s: float = PUBLISH_WAIT_S,
    ):
        if policy not in ("finish_drop", "drop"):
            raise ValueError("store_full_policy must be finish_drop or drop")
        if save_mode not in ("per_layer", "end"):
            raise ValueError("kv.direct_save_mode must be per_layer or end")
        self.sdk = sdk
        self.preload_pool = preload_pool if preload_pool is not None else pool
        self.store_pool = store_pool if store_pool is not None else pool
        self.pool = self.store_pool
        self.native = native
        self.geometry = geometry
        self.policy = policy
        self.save_mode = save_mode
        self.board = board
        self._preload_bytes = preload_bytes
        self._store_bytes = store_bytes
        self.queue = DramPipelineQueue(lookup_limit)
        self.queue.set_capacity(self._queue_need, self._queue_free)
        self.publish_wait_s = max(0.0, float(publish_wait_s))
        self._publish_wait_since: dict[tuple[str, int], float] = {}
        self._mu = threading.Lock()
        self._reqs: dict[str, _Req] = {}
        self._stop = threading.Event()
        self._wake_lookup = threading.Event()
        self._wake_preload = threading.Event()
        self._wake_h2d = threading.Event()
        self._wake_d2h = threading.Event()
        self._wake_store = threading.Event()
        self._wake_commit = threading.Event()
        self._notes = []
        self._storing = None
        self._lookup_backoff: dict[str, int] = {}
        self._preload_backoff: dict[str, int] = {}
        self._snaps = []
        self._d2h_handed = set()
        self._bind_pool_wakes()
        self._pending_alloc: dict[str, tuple[list, int]] = {}
        self._store = []
        self._jobs = {}
        self._park_wait: dict[str, int] = {}
        self._park_submitting: set[str] = set()
        self._store_move: set[str] = set()
        self._seq = 1
        self._recving = set()
        self._failed = set()
        self._sending = set()
        self._sending_snapshot = set()
        self._recving_snapshot = set()
        self._failed_snapshot = set()
        self._slots: dict[str, int] = {}
        self._engine_finished: set[str] = set()
        self.drops = []
        self.stale_drops = 0
        self.h2d_skip_empty = 0
        self.h2d_skip_geometry = 0
        self.h2d_skip_full = 0
        self._lookup_inflight = 0
        self._lookup_busy = 0
        self._phase_diag_ms = 0
        self._layer_bases: dict[str, int] = {}
        self._layer_events: dict = {}
        self._base_epoch = 0
        # The event recorded at the start of the current wait_for_save, end mode only.
        self._save_event = None
        # Requests wait_for_save still has to materialize, refresh, or hand off.
        self._save_active: set[str] = set()
        self._scan_observe_ms = 0.0
        self._scan_observe_reqs = 0
        self._scan_save_ms = 0.0
        self._scan_save_reqs = 0
        self._scan_open_blocks = 0
        self._scan_event_create_ms = 0.0
        self._scan_event_record_ms = 0.0
        self._scan_save_layer_ms = 0.0
        self._threads = []
        self._started = False
        self._pin_logged = False
        self._torch = None
        self._compute_stream = None
        self._store_stream = None
        self._d2h_inflight = []
        self.d2h_copies = []

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        targets = (
            ("lookup", self._lookup_loop),
            ("preload", self._preload_loop),
            ("h2d", self._h2d_loop),
            ("d2h", self._d2h_loop),
            ("store", self._store_loop),
            ("commit", self._commit_loop),
            ("storeGC", self._gc_loop),
        )
        wake = getattr(self.sdk, "set_commit_wake", None)
        if callable(wake):
            wake(self._wake_commit.set)
        for name, fn in targets:
            thread = threading.Thread(target=fn, name="surdata-" + name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._phase_diag_ms = load_direct_phase_diag_ms()
        if self._phase_diag_ms > 0:
            thread = threading.Thread(
                target=self._phase_diag_loop, name="surdata-phase-diag", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def shutdown(self) -> None:
        self._stop.set()
        events = [
            self._wake_lookup,
            self._wake_preload,
            self._wake_h2d,
            self._wake_d2h,
            self._wake_store,
            self._wake_commit,
        ]
        for pool in (self.preload_pool, self.store_pool):
            if pool is None:
                continue
            events.append(pool.watermark_event())
            events.append(pool.free_event())
        for event in events:
            event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)

    def set_geometry(self, geometry: KvGeometry) -> None:
        self.geometry = geometry
        if hasattr(self.sdk, "geometry"):
            self.sdk.geometry = geometry
        self._ensure_pools()

    def _ensure_pools(self) -> None:
        if self.geometry is None:
            return
        slot = self.geometry.kvb_bytes(self.geometry.block_size)
        if slot <= 0:
            return
        if self.preload_pool is None and self._preload_bytes:
            self.preload_pool = self._open_pinned_pool(
                slot, int(self._preload_bytes), "preload", on_free=self._wake_preload.set
            )
        if self.store_pool is None and self._store_bytes:
            self.store_pool = self._open_pinned_pool(
                slot,
                int(self._store_bytes),
                "store",
                watermark_bytes=int(load_external_kv_worker_count()) * int(slot),
            )
            self.pool = self.store_pool
        self._bind_pool_wakes()

    def _bind_pool_wakes(self) -> None:
        preload = self.preload_pool
        store = self.store_pool
        if preload is not None and preload is not store:
            preload.set_on_free(self._wake_preload.set)
        if store is not None and store is not preload:
            store.set_on_free(self._wake_d2h.set)
        elif store is not None:
            preload_wake = self._wake_preload.set
            d2h_wake = self._wake_d2h.set

            def _both() -> None:
                preload_wake()
                d2h_wake()

            store.set_on_free(_both)

    def _pin_line(self, kind: str, mapped: int, pin_rc, check: str) -> None:
        mode = "a" if self._pin_logged else "w"
        self._pin_logged = True
        line = "kind=%s alloc=anon-private mlock=1 pin_rc=%s check=%s bytes=%s\n" % (
            kind,
            pin_rc,
            check,
            int(mapped),
        )
        try:
            with open("/tmp/surdata_pin.txt", mode, encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass
        print("SURDATA_PIN %s" % line.strip(), file=sys.stderr, flush=True)

    def _open_pinned_pool(self, slot: int, capacity: int, kind: str, on_free=None, watermark_bytes: int = 0):
        """Allocate, register, and prove the copy for this direction is async."""
        native = self.native
        alloc = getattr(native, "hostmem_alloc", None) if native is not None else None
        release = getattr(native, "hostmem_free", None) if native is not None else None
        if not callable(alloc) or not callable(release):
            raise RuntimeError("hostmem_alloc missing")
        try:
            base, mapped = alloc(int(capacity))
        except Exception:
            _LOG.error("%s pool alloc failed: %s bytes", kind, capacity, exc_info=True)
            raise
        base = int(base)
        mapped = int(mapped)
        from surdata_vllm.store_pool import StorePool

        try:
            pool = StorePool(
                slot,
                capacity_bytes=int(capacity),
                kind=kind,
                on_free=on_free,
                watermark_bytes=int(watermark_bytes) if kind == "store" else 0,
                base_ptr=base,
                mapped_bytes=mapped,
                release=release,
            )
        except Exception:
            if callable(release):
                release(base, mapped)
            raise
        if kind == "preload":
            self.preload_pool = pool
        else:
            self.store_pool = pool
            self.pool = pool
        direction, label = (2, "d2h") if kind == "store" else (1, "h2d")
        pin = getattr(native, "host_pin", None)
        check = getattr(native, "copy_async_check", None)
        try:
            if not callable(pin):
                raise RuntimeError("host_pin missing")
            pin_rc = int(pin(base, mapped))
        except Exception:
            _LOG.error("%s pool pin failed: %s bytes", kind, mapped, exc_info=True)
            self._pin_line(kind, mapped, "exc", "%s-sync rc=exc" % label)
            pool.close()
            raise
        if pin_rc != 0:
            _LOG.error("%s pool pin failed: rc=%s bytes=%s", kind, pin_rc, mapped)
            self._pin_line(kind, mapped, pin_rc, "%s-sync rc=%s" % (label, pin_rc))
            pool.close()
            raise RuntimeError("cudaHostRegister failed rc=%s bytes=%s" % (pin_rc, mapped))
        try:
            if not callable(check):
                raise RuntimeError("copy_async_check missing")
            check_rc = int(check(base, mapped, direction))
        except Exception:
            _LOG.error("%s pool async check failed: %s bytes", kind, mapped, exc_info=True)
            self._pin_line(kind, mapped, 0, "%s-sync rc=exc" % label)
            pool.close()
            raise
        if check_rc != 0:
            _LOG.error("%s pool async check failed: rc=%s bytes=%s", kind, check_rc, mapped)
            self._pin_line(kind, mapped, 0, "%s-sync rc=%s" % (label, check_rc))
            pool.close()
            raise RuntimeError("copy async check failed rc=%s bytes=%s" % (check_rc, mapped))
        self._pin_line(kind, mapped, 0, "%s-async" % label)
        unpin = getattr(native, "host_unpin", None)
        pool.note_pinned(unpin if callable(unpin) else None)
        return pool

    def on_new_request(self, req_id: str, tokens: list) -> int:
        key = str(req_id)
        tokens = [int(tok) for tok in tokens]
        current = self.queue.get(key)
        generation = 1 if current is None else int(current.generation)
        self._ensure(key, generation, tokens)
        self.queue.enqueue_lookup(
            key, {"tokens": tokens}, len(tokens), generation=generation, slot=self._slots.get(key)
        )
        self._wake_lookup.set()
        return generation

    def ingest(self, meta: ConnectorMeta) -> None:
        if meta is None:
            return
        for item in meta.preempted:
            self._bind_slot(item)
            self.preempt(item["req_id"])
        for item in meta.lookups:
            key = str(item["req_id"])
            tokens = [int(tok) for tok in item["tokens"]]
            generation = int(item["generation"])
            self._ensure(key, generation, tokens, item.get("slot"))
            self.queue.enqueue_lookup(
                key, {"tokens": tokens}, len(tokens), generation=generation, slot=item.get("slot")
            )
            self._wake_lookup.set()
        for item in meta.allocs:
            self._bind_slot(item)
            self.update_state_after_alloc(item["req_id"], item["block_ids"], item["external_tokens"])
        for item in meta.finished:
            self.request_finished(
                item["req_id"],
                item["block_ids"],
                item["num_tokens"],
                slot=item.get("slot"),
                generation=item.get("generation"),
            )
        for item in meta.save_hints:
            self._bind_slot(item)
            self.ingest_blocks(item["req_id"], item["generation"], item["block_ids"], item["num_tokens"])

    def _bind_slot(self, item) -> None:
        if not isinstance(item, dict) or item.get("slot") is None:
            return
        slot = int(item["slot"])
        if slot < 0:
            return
        key = str(item.get("req_id", ""))
        if not key:
            return
        self._slots[key] = slot
        with self._mu:
            req = self._reqs.get(key)
            if req is not None:
                req.slot = slot

    def _ensure(self, req_id: str, generation: int, tokens: list, slot=None) -> _Req:
        req_id = str(req_id)
        tokens = list(tokens)
        if slot is not None and int(slot) >= 0:
            self._slots[req_id] = int(slot)
        note = getattr(self.sdk, "note_tokens", None)
        if callable(note):
            note(req_id, tokens)
        abandon = None
        with self._mu:
            current = self._reqs.get(req_id)
            if current is not None and current.generation == generation and not current.closed:
                current.tokens = list(tokens)
                if req_id in self._slots:
                    current.slot = self._slots[req_id]
                self._take_pending_alloc_locked(current)
                self._sync_save_active_locked(current)
                return current
            if current is not None and int(current.generation) != int(generation):
                abandon = (req_id, int(current.generation), "abandon", 0)
                self._notes.append(abandon)
            owned = int(self._slots.get(req_id, -1))
            req = _Req(
                req_id=req_id,
                generation=generation,
                tokens=list(tokens),
                num_tokens=len(tokens),
                slot=owned,
            )
            self._reqs[req_id] = req
            self._take_pending_alloc_locked(req)
            self._sync_save_active_locked(req)
        if abandon is not None:
            self._wake_store.set()
        return req

    def _queue_need(self, entry) -> int:
        geo = self.geometry
        if geo is None:
            return 0
        tokens = int(getattr(entry, "prompt_tokens", 0) or 0)
        if tokens <= 0:
            return 0
        block = max(1, int(geo.block_size))
        blocks = max(1, (int(tokens) + block - 1) // block)
        need = int(geo.kvb_bytes(geo.block_size)) * blocks
        pool = self.preload_pool
        if pool is None:
            return need
        capacity = int(getattr(pool, "slot_count", 0)) * int(getattr(pool, "slot_bytes", 0))
        if capacity and need > capacity:
            return 0
        return need

    def _queue_free(self) -> int:
        pool = self.preload_pool
        if pool is None:
            return 1 << 62
        return int(pool.ledger_free())

    def preempt(self, req_id: str) -> None:
        """Unfinished KV already on GPU is copied into a new preload allocation.

        A request that never received GPU blocks is closed. One that is still
        only on the wait ring is removed from that ring. D2H uses the block
        ids from this call. While this worker is running, return only after
        that copy has left ``_jobs``.
        """
        key = str(req_id)
        generation = 0
        park = False
        waiting = False
        with self._mu:
            req = self._reqs.get(key)
            if req is None or req.closed or req.finished:
                waiting = True
            elif req.block_ids:
                park = True
                generation = int(req.generation)
                req.phase = PHASE_READING
                self._park_wait[key] = generation
            else:
                waiting = True
        if waiting and not park:
            entry = self.queue.get(key)
            if entry is not None and entry.status == STATUS_LOOKUP_PENDING and not entry.lookup_claimed:
                self.queue.drop_wait(key)
                return
            self.cancel(key)
            return
        self._publish(key, generation, PHASE_READING, 0)
        self._drain_parks()
        held = bool(getattr(self.native, "poll_hold", False))
        if not self._stop.is_set() and not held:
            while self._park_in_jobs(key):
                self._poll_copies()
        self._wake_h2d.set()

    def _park_in_jobs(self, req_id: str) -> bool:
        key = str(req_id)
        for kind in list(self._jobs.values()):
            if kind and kind[0] == CopyKind.PARK and str(kind[1]) == key:
                return True
        return False

    def cancel(self, req_id: str) -> None:
        key = str(req_id)
        self._park_wait.pop(key, None)
        self.queue.remove(key)
        for item in list(self._publish_wait_since):
            if item[0] == key:
                self._publish_wait_since.pop(item, None)
        slots = None
        missing = False
        with self._mu:
            req = self._reqs.get(key)
            if req is None:
                missing = True
            else:
                req.closed = True
                req.phase = PHASE_CLOSED
                generation = req.generation
                slots = getattr(req, "_preload_slots", None)
                req._preload_slots = None
                self._notes.append((key, int(generation), "abandon", 0))
                self._sync_save_active_locked(req)
        if missing:
            self._wake_preload.set()
            self._cancel_inflight_h2d(key)
            return
        if slots and self.preload_pool is not None:
            self.preload_pool.abort_unstored(slots)
        self._publish(key, generation, PHASE_CLOSED, 0, 1)
        self._wake_lookup.set()
        self._wake_preload.set()
        self._wake_store.set()
        self._cancel_inflight_h2d(key)
        self._cancel_inflight_kind(key, CopyKind.PARK)

    def _cancel_native_job(self, job) -> None:
        cancel_fn = getattr(self.native, "request_cancel", None)
        if callable(cancel_fn) and isinstance(job, dict):
            cancel_fn(job)

    def _cancel_inflight_h2d(self, req_id: str) -> None:
        self._cancel_inflight_kind(req_id, CopyKind.H2D)

    def _cancel_inflight_kind(self, req_id: str, direction) -> None:
        key = str(req_id)
        jobs = []
        for kind in list(self._jobs.values()):
            if not kind or kind[0] != direction or str(kind[1]) != key:
                continue
            job = kind[-1]
            if isinstance(job, dict):
                jobs.append(job)
        for job in jobs:
            self._cancel_native_job(job)

    def matched_tokens(self, req_id: str, computed: int = 0):
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is None or req.closed:
                return None, False
            return matched_from_phase(req.phase, req.hit_tokens, computed)

    def _take_pending_alloc_locked(self, req: _Req) -> None:
        """Caller holds ``_mu``. A alloc that arrived before the request is applied once."""
        pending = self._pending_alloc.pop(req.req_id, None)
        if not pending:
            return
        ids, external = pending
        req.external_tokens = int(external)
        if req.full_block_table:
            return
        if ids:
            req.block_ids = [int(block) for block in ids]
            req.full_block_table = False

    def update_state_after_alloc(self, req_id: str, block_ids: list, external_tokens: int) -> bool:
        release = False
        ids = [int(block) for block in block_ids]
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is None or req.closed:
                if ids:
                    self._pending_alloc[str(req_id)] = (ids, int(external_tokens))
                return False
            req.external_tokens = int(external_tokens)
            if not req.full_block_table:
                req.block_ids = ids
                req.full_block_table = False
            self._sync_save_active_locked(req)
            # A miss already has its KV once the scheduler assigns GPU blocks.
            if req.phase == PHASE_MISS:
                release = True
        if release and self.queue.release_lookup_window(str(req_id)):
            self._wake_lookup.set()
        if ids:
            self._wake_h2d.set()
        if int(external_tokens) <= 0:
            return False
        return True

    def request_finished(self, req_id: str, block_ids: list, num_tokens: int, slot=None, generation=None) -> bool:
        """Free every block that is not already on the device, then the set once.

        vLLM frees the whole GPU set when this id appears in ``finished_sending``.
        A block still sitting in ``_snaps``, or not queued at all, is dropped here.
        A block already in ``_d2h_handed`` or ``_d2h_inflight`` stays until the
        D2H thread's own copy event. That wait does not hold any other request.
        """
        key = str(req_id)
        release_slot = -1
        release_gen = 0
        drops = []
        submits = []
        with self._mu:
            req = self._reqs.get(key)
            if req is not None and (
                req.closed or (generation is not None and int(req.generation) != int(generation))
            ):
                return False
            if req is not None and req.finished:
                return True
            if slot is not None and int(slot) >= 0:
                self._slots[key] = int(slot)
            if req is None:
                req = _Req(
                    req_id=key,
                    generation=0 if generation is None else int(generation),
                    tokens=[],
                    slot=int(self._slots.get(key, -1)),
                )
                self._reqs[key] = req
            elif key in self._slots:
                req.slot = self._slots[key]
            req.finished = True
            req.num_tokens = int(num_tokens)
            if block_ids:
                req.block_ids = [int(block) for block in block_ids]
            self._materialize_locked(req, None)
            park_gpu = self._claim_waiting_park_locked(req)
            if park_gpu:
                for block in req.blocks.values():
                    if not block.d2h_done and not block.saved:
                        self._d2h_handed.add((key, int(req.generation), int(block.block_id)))
            drops, submits = self._collect_finished_settle_locked(req)
            if submits:
                self._snaps.append(submits)
            release_slot = int(self._slots.get(key, req.slot))
            release_gen = int(req.generation if generation is None else generation)
            self._notes.append((key, int(req.generation), "finished", 0))
            park_copy = None
            if park_gpu:
                park_copy = (
                    list(req.block_ids),
                    self._copy_layer_bases_locked(req),
                    int(req.generation),
                    int(req.num_tokens or 0),
                )
            self._sync_save_active_locked(req)
        for block in drops:
            self._drop_block(key, release_gen, block)
        if submits and self.store_pool is not None:
            level, span = self._watermark_locked(submits)
            self.store_pool.request_watermark(level, span)
        self._wake_store.set()
        if submits:
            self._wake_d2h.set()
        if self.board is not None and release_slot >= 0 and release_gen > 0:
            self.board.release(release_slot, release_gen)
        if park_copy:
            self._submit_finished_store(key, park_copy)
        return True

    def _claim_waiting_park_locked(self, req: _Req) -> bool:
        """A preload D2H that has not been queued becomes a store-pool copy."""
        key = str(req.req_id)
        if key not in self._park_wait:
            return False
        if key in self._park_submitting or self._park_in_jobs(key):
            return False
        self._park_wait.pop(key, None)
        self._store_move.add(key)
        return True

    def _drop_store_move(self, req_id: str, generation: int) -> None:
        key = str(req_id)
        with self._mu:
            self._store_move.discard(key)
            req = self._reqs.get(key)
            if req is None or int(req.generation) != int(generation):
                return
            for block in list(req.blocks.values()):
                self._forget_handed_locked(req, block.block_id)
            self._release_saved_locked(req)

    def _submit_finished_store(self, req_id: str, copy) -> None:
        """Copy GPU blocks of a finished request into the store pool."""
        block_ids, layer_bases, generation, tokens = copy
        key = str(req_id)
        slots = None
        job = None
        try:
            if (
                not block_ids
                or self.geometry is None
                or self.store_pool is None
                or len(layer_bases) < self.geometry.num_layers
            ):
                self._drop_store_move(key, generation)
                return
            count = len(block_ids)
            payload = self.geometry.layer_payload(self.geometry.block_size)
            if int(tokens) <= 0:
                tokens = count * int(self.geometry.block_size)
            need = payload * count * len(layer_bases)
            slots = self.store_pool.try_allocate(need, "store", key, generation)
            if not slots:
                self._drop_store_move(key, generation)
                return
            spans = self._park_spans(slots, block_ids, layer_bases, payload)
            job = self._job(key, generation)
            self._jobs[job["sequence"]] = (CopyKind.D2H, key, int(generation), slots, int(tokens), job)
            rc = int(self.native.submit_d2h(job, spans, generation, generation, []) or 0)
        except Exception:
            rc = -1
        if rc != 0:
            if job is not None:
                self._jobs.pop(job["sequence"], None)
            if slots and self.store_pool is not None:
                self.store_pool.abort_unstored(slots)
            self._drop_store_move(key, generation)

    def _move_park_bytes_to_store(self, req_id, generation, slots, tokens: int) -> None:
        """The preload copy already finished. Keep the bytes in the store pool."""
        import ctypes

        key = str(req_id)
        store = self.store_pool
        if store is None or not slots:
            if slots and self.preload_pool is not None:
                self.preload_pool.abort_unstored(slots)
            self._drop_store_move(key, generation)
            return
        need = self._slot_bytes(slots)
        store_slots = store.try_allocate(need, "store", key, generation)
        if not store_slots:
            if self.preload_pool is not None:
                self.preload_pool.abort_unstored(slots)
            self._drop_store_move(key, generation)
            return
        for src, dst in zip(slots, store_slots):
            nbytes = min(int(src.nbytes), int(dst.nbytes))
            if nbytes > 0 and int(src.buf) and int(dst.buf):
                ctypes.memmove(int(dst.buf), int(src.buf), nbytes)
        if self.preload_pool is not None:
            self.preload_pool.abort_unstored(slots)
        self._enqueue_store_save(key, generation, store_slots, int(tokens))
        with self._mu:
            req = self._reqs.get(key)
            if req is None or int(req.generation) != int(generation):
                return
            for block in list(req.blocks.values()):
                block.d2h_done = True
                self._forget_handed_locked(req, block.block_id)
            self._release_saved_locked(req)

    def _enqueue_store_save(self, req_id: str, generation: int, slots, tokens: int) -> None:
        block_tokens = 1 if self.geometry is None else int(self.geometry.block_size)
        try:
            rc = self.sdk.enqueue_save(
                req_id,
                generation,
                int(tokens),
                block_tokens,
                slots,
                0,
                lambda status, req_id=req_id, generation=generation, slots=slots: self._on_stored(
                    req_id, generation, slots, status
                ),
            )
        except Exception:
            rc = -1
        self._store_move.discard(str(req_id))
        if rc != 0 and self.store_pool is not None:
            self.store_pool.abort_unstored(slots)
            self._drop_store_move(req_id, generation)

    def _finish_store_copy(self, kind, item) -> None:
        _, req_id, generation, slots, tokens = kind[:5]
        if int(item.get("status", 0)) != 0:
            if slots and self.store_pool is not None:
                self.store_pool.abort_unstored(slots)
            self._drop_store_move(req_id, generation)
            return
        self._enqueue_store_save(str(req_id), int(generation), slots, int(tokens))
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is None or int(req.generation) != int(generation):
                return
            for block in list(req.blocks.values()):
                block.d2h_done = True
                self._forget_handed_locked(req, block.block_id)
            self._release_saved_locked(req)

    def _device_copy_ids_locked(self, req: _Req) -> set:
        """Block ids whose memcpy has left the queue. GPU memory is still the source."""
        owned = set()
        for item in self._d2h_inflight:
            if str(item.get("req_id")) != req.req_id:
                continue
            if int(item.get("generation", -1)) != int(req.generation):
                continue
            block = item.get("block")
            if block is not None:
                owned.add(int(block.block_id))
        for req_id, generation, block_id in self._d2h_handed:
            if str(req_id) == req.req_id and int(generation) == int(req.generation):
                owned.add(int(block_id))
        return owned

    def _detach_queued_locked(self, req: _Req, block_id: int) -> None:
        """Drop one block from ``_snaps``. The device has not started this copy."""
        kept = []
        for snap in self._snaps:
            stay = []
            for req_id, generation, block in snap:
                if (
                    str(req_id) == req.req_id
                    and int(generation) == int(req.generation)
                    and int(block.block_id) == int(block_id)
                ):
                    continue
                stay.append((req_id, generation, block))
            if stay:
                kept.append(stay)
        self._snaps = kept

    def _mark_block_done_locked(self, req: _Req, block: _Block) -> None:
        block.saved = True
        block.inflight = False
        block.d2h_done = True
        self._untrack_open_locked(req, block.block_id)
        self._forget_handed_locked(req, block.block_id)
        self._release_saved_locked(req)

    def _block_past_tokens(self, req: _Req, block: _Block) -> bool:
        block_size = 0 if self.geometry is None else int(self.geometry.block_size)
        if block_size <= 0:
            return False
        return int(block.token_index) * block_size >= int(req.num_tokens)

    def _hit_covers_locked(self, req: _Req, block: _Block) -> bool:
        """True when this block's tokens are already inside the prefix hit.

        A hit may end inside a block. That tail is covered too, so it is not
        copied again. ``hit_tokens // block_size`` would leave the tail out.
        """
        hit = int(req.hit_tokens)
        if hit <= 0 or self.geometry is None:
            return False
        block_size = int(self.geometry.block_size)
        if block_size <= 0:
            return False
        start = int(block.token_index) * block_size
        end = start + int(block.valid_tokens)
        return end > start and end <= hit

    def _queued_block_ids_locked(self, req: _Req) -> set:
        """Block ids sitting in ``_snaps`` that D2H has not taken yet."""
        found = set()
        key = str(req.req_id)
        generation = int(req.generation)
        for snap in self._snaps:
            for req_id, gen, block in snap:
                if str(req_id) == key and int(gen) == generation:
                    found.add(int(block.block_id))
        return found

    def _d2h_busy_locked(self, req: _Req) -> bool:
        """True while this generation still has a D2H copy or a save queued."""
        key = str(req.req_id)
        generation = int(req.generation)
        for snap in self._snaps:
            for req_id, gen, _block in snap:
                if str(req_id) == key and int(gen) == generation:
                    return True
        for req_id, gen, _block_id in self._d2h_handed:
            if str(req_id) == key and int(gen) == generation:
                return True
        for item in self._d2h_inflight:
            if str(item.get("req_id")) == key and int(item.get("generation", -1)) == generation:
                return True
        for item in self._store:
            if str(item[0]) == key and int(item[1]) == generation:
                return True
        for kind in self._jobs.values():
            if not kind or str(kind[1]) != key:
                continue
            if kind[0] in (CopyKind.PARK, CopyKind.D2H):
                return True
        if key in self._store_move:
            return True
        req = self._reqs.get(key)
        if req is not None and int(req.generation) == generation:
            for block in req.blocks.values():
                if block.inflight and not block.saved and not block.d2h_done:
                    return True
        return False

    def _forget_handed_locked(self, req: _Req, block_id: int) -> None:
        self._d2h_handed.discard((str(req.req_id), int(req.generation), int(block_id)))

    def _collect_finished_settle_locked(self, req: _Req):
        """Caller holds ``_mu``. Queue every block that has not started copying.

        Full blocks and the short tail both go to ``submits``. A copy already in
        ``_snaps``, ``_d2h_handed``, or ``_d2h_inflight`` stays there.
        ``inflight`` with none of those records is a lost copy and drops.
        Blocks past ``num_tokens`` drop. A prefix hit that already covers the
        block, including a tail inside the hit, is sealed without another copy.
        """
        drops = []
        submits = []
        if not req.finished or req.sending_published:
            return drops, submits
        on_device = self._device_copy_ids_locked(req)
        queued = self._queued_block_ids_locked(req)
        park_reading = self._park_in_jobs(req.req_id) or str(req.req_id) in self._park_submitting
        for block in list(req.blocks.values()):
            if block.d2h_done or block.saved:
                continue
            bid = int(block.block_id)
            if park_reading or bid in on_device or bid in queued:
                continue
            if block.inflight:
                drops.append(block)
                continue
            if self._block_past_tokens(req, block):
                drops.append(block)
                continue
            if self._hit_covers_locked(req, block):
                self._mark_block_done_locked(req, block)
                continue
            if self.save_mode == "end" and self._save_event is not None:
                block.save_event = self._save_event
            block.inflight = True
            self._untrack_open_locked(req, bid)
            submits.append((req.req_id, int(req.generation), block))
        self._release_saved_locked(req)
        return drops, submits

    def _release_saved_locked(self, req: _Req) -> None:
        if req.sending_published or req.req_id in self._sending_snapshot:
            self._sync_save_active_locked(req)
            return
        if req.finished and not self._d2h_busy_locked(req):
            self._sending.add(req.req_id)
        self._sync_save_active_locked(req)

    def _detach_past_tokens_locked(self, req: _Req) -> list:
        """Blocks past ``num_tokens`` that have not started a device copy."""
        drops = []
        on_device = self._device_copy_ids_locked(req)
        for block in list(req.blocks.values()):
            if not self._block_past_tokens(req, block):
                continue
            if block.d2h_done or block.saved:
                continue
            if int(block.block_id) in on_device:
                continue
            self._detach_queued_locked(req, int(block.block_id))
            drops.append(block)
        return drops

    def ingest_blocks(self, req_id: str, generation: int, block_ids: list, num_tokens: int) -> None:
        ids = [int(block) for block in block_ids]
        drops = []
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is None or req.closed or int(req.generation) != int(generation):
                return
            req.block_ids = ids
            req.num_tokens = int(num_tokens)
            req.full_block_table = bool(ids)
            self._materialize_locked(req, ids)
            if req.finished:
                drops = self._detach_past_tokens_locked(req)
                self._release_saved_locked(req)
            self._sync_save_active_locked(req)
        for block in drops:
            self._drop_block(str(req_id), int(generation), block)
        if ids:
            self._wake_h2d.set()

    def _layer_src_locked(self, req: _Req):
        """Prefer the global table once it covers every layer.

        A request copy is only the fallback. The hot path no longer refreshes
        that copy, so preferring it would keep H2D and park on a stale base.
        """
        need = 0 if self.geometry is None else int(self.geometry.num_layers)
        if need > 0 and len(self._layer_bases) >= need:
            return self._layer_bases, self._layer_events, int(self._base_epoch)
        if req.layer_base:
            return req.layer_base, req.layer_events, int(req.base_epoch)
        return self._layer_bases, self._layer_events, int(self._base_epoch)

    def _layer_epoch_locked(self, req: _Req) -> int:
        _bases, _events, epoch = self._layer_src_locked(req)
        return int(epoch)

    def _copy_layer_bases_locked(self, req: _Req) -> dict:
        bases, _events, _epoch = self._layer_src_locked(req)
        return dict(bases)

    def _token_limit_locked(self, req: _Req) -> int:
        return int(req.num_tokens or len(req.tokens) or 0)

    def _finish_handoff_pending_locked(self, req: _Req) -> bool:
        """True while wait_for_save still has to publish finished_sending."""
        if not req.finished or req.sending_published:
            return False
        if req.req_id in self._sending or req.req_id in self._sending_snapshot:
            return False
        return self._d2h_busy_locked(req)

    def _wants_save_scan_locked(self, req: _Req) -> bool:
        if req.closed:
            return False
        if req.open_blocks:
            return True
        if req.block_ids and int(req.built_tokens) < self._token_limit_locked(req):
            # A finished request that is not copying is skipped before materialize.
            # Keeping it here would scan it on every later forward.
            if not req.finished or self._d2h_busy_locked(req):
                return True
        return self._finish_handoff_pending_locked(req)

    def _sync_save_active_locked(self, req: _Req) -> None:
        if self._wants_save_scan_locked(req):
            self._save_active.add(req.req_id)
        else:
            self._save_active.discard(req.req_id)

    def _flush_scan_trace(self) -> None:
        """Append this forward's scan counters to the open decode-trace line."""
        if not _decode_trace_enabled():
            return
        suffix = (
            "observe_ms=%.3f observe_reqs=%d save_scan_ms=%.3f save_scan_reqs=%d "
            "save_open_blocks=%d event_create_ms=%.3f event_record_ms=%.3f save_layer_ms=%.3f"
            % (
                self._scan_observe_ms,
                self._scan_observe_reqs,
                self._scan_save_ms,
                self._scan_save_reqs,
                self._scan_open_blocks,
                self._scan_event_create_ms,
                self._scan_event_record_ms,
                self._scan_save_layer_ms,
            )
        )
        self._scan_observe_ms = 0.0
        self._scan_observe_reqs = 0
        self._scan_save_ms = 0.0
        self._scan_save_reqs = 0
        self._scan_open_blocks = 0
        self._scan_event_create_ms = 0.0
        self._scan_event_record_ms = 0.0
        self._scan_save_layer_ms = 0.0
        if not _attach_step_suffix(suffix):
            print("SURDATA_DECODE_TRACE " + suffix, file=sys.stderr, flush=True)

    def set_layer_bases(self, bases: dict) -> None:
        """HBM bases from register_kv_caches. H2D can run before the first forward."""
        with self._mu:
            new = {str(name): int(ptr) for name, ptr in bases.items()}
            if new != self._layer_bases:
                self._base_epoch += 1
            self._layer_bases = new
        self._wake_h2d.set()

    def observe_layer(self, layer_name: str, base_ptr: int, event) -> None:
        """Record this layer's base and completion event once for every request."""
        trace = _decode_trace_enabled()
        started = time.perf_counter() if trace else 0.0
        with self._mu:
            name = str(layer_name)
            ptr = int(base_ptr)
            if self._layer_bases.get(name) != ptr:
                self._base_epoch += 1
            self._layer_bases[name] = ptr
            self._layer_events[name] = event
        self._wake_h2d.set()
        if trace:
            self._scan_observe_ms += (time.perf_counter() - started) * 1000.0

    def note_save_layer_trace(self, create_ms: float, record_ms: float, save_ms: float) -> None:
        """Accumulate this forward's save_kv_layer pieces. save_ms includes observe."""
        if not _decode_trace_enabled():
            return
        self._scan_event_create_ms += float(create_ms)
        self._scan_event_record_ms += float(record_ms)
        self._scan_save_layer_ms += float(save_ms)

    def note_block(self, req_id: str, generation: int, block_id: int, valid_tokens: int, src: dict, events: dict, token_index: int = 0) -> None:
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is None or req.closed or int(req.generation) != int(generation):
                return
            stride = 0 if self.geometry is None else self.geometry.layer_stride()
            block = _Block(
                block_id=int(block_id),
                valid_tokens=int(valid_tokens),
                src=dict(src),
                events=dict(events),
                token_index=int(token_index),
                stride=stride,
                src_epoch=self._layer_epoch_locked(req),
            )
            req.blocks[int(block_id)] = block
            self._track_open_locked(req, block)
            self._sync_save_active_locked(req)

    def wait_for_save(self, next_bytes: int = 0) -> None:
        """Seal one snapshot and wake D2H and storeGC. No storage and no sync.

        ``need`` is the sum of ``kvb_bytes`` for the full blocks this call is
        about to hand to D2H. ``span`` is the largest of those, or 0 when
        there are none. The store pool publishes ``max(watermark_bytes, need)``.
        Preload is not woken here. ``next_bytes`` is ignored.
        """
        del next_bytes
        self.note_compute_stream()
        if self.save_mode == "end":
            self._save_event = self._record_tail_event()
        tid = threading.get_ident()
        before_sdk = _calls(self.sdk, tid)
        before_submit = _calls(self.native, tid)
        before_sync = int(getattr(self.native, "syncs", 0))
        pending_drops = []
        trace = _decode_trace_enabled()
        scan_started = time.perf_counter() if trace else 0.0
        scan_reqs = 0
        scan_open = 0
        with self._mu:
            blocks = []
            for req_id in list(self._save_active):
                req = self._reqs.get(req_id)
                if req is None or req.closed:
                    self._save_active.discard(req_id)
                    continue
                if trace:
                    scan_reqs += 1
                    scan_open += len(req.open_blocks)
                self._materialize_locked(req, None)
                if req.finished:
                    drops, submits = self._collect_finished_settle_locked(req)
                    pending_drops.extend((req.req_id, int(req.generation), block) for block in drops)
                    blocks.extend(submits)
                still = []
                for block in req.open_blocks:
                    if self._keep_open_locked(req, block, blocks):
                        still.append(block)
                req.open_blocks = still
                self._sync_save_active_locked(req)
            self._snaps.append(blocks)
            level, span = self._watermark_locked(blocks)
        if trace:
            self._scan_save_ms += (time.perf_counter() - scan_started) * 1000.0
            self._scan_save_reqs += scan_reqs
            self._scan_open_blocks += scan_open
        for req_id, generation, block in pending_drops:
            self._drop_block(req_id, generation, block)
        if self.store_pool is not None:
            self.store_pool.request_watermark(level, span)
        self._wake_d2h.set()
        self._wake_commit.set()
        if _calls(self.sdk, tid) != before_sdk or _calls(self.native, tid) != before_submit:
            raise RuntimeError("wait_for_save called storage or a copy")
        if int(getattr(self.native, "syncs", 0)) != before_sync:
            raise RuntimeError("wait_for_save synchronized the device")
        self._flush_scan_trace()

    def save_layer_event(self, layer_name: str, base_ptr: int, event) -> None:
        tid = threading.get_ident()
        before_sdk = _calls(self.sdk, tid)
        before_submit = _calls(self.native, tid)
        self.observe_layer(layer_name, base_ptr, event)
        if _calls(self.sdk, tid) != before_sdk or _calls(self.native, tid) != before_submit:
            raise RuntimeError("save_kv_layer called storage or a copy")

    def _handoff_sending_locked(self) -> set:
        """Ids to hand the scheduler now.

        Overlap with ``_recving`` is not returned: the scheduler frees a
        finished request on the recving path. Those ids are marked published
        and dropped here so the next step cannot report them as sending.
        The snapshot is only this copy. Ids that arrive before
        ``clear_published`` stay in ``_sending``.
        """
        overlap = self._sending & self._recving
        for req_id in overlap:
            req = self._reqs.get(req_id)
            if req is not None:
                req.sending_published = True
            self._sending.discard(req_id)
        handed = set(self._sending)
        self._sending_snapshot = set(handed)
        return handed

    def transfer_results(self, finished_req_ids=None) -> TransferResults:
        self._poll_copies()
        out = TransferResults()
        with self._mu:
            if finished_req_ids:
                self._engine_finished.update(str(item) for item in finished_req_ids)
            out.finished_sending = self._handoff_sending_locked()
            self._recving_snapshot = set(self._recving)
            self._failed_snapshot = set(self._failed)
            out.finished_recving = set(self._recving_snapshot)
            out.failed_recving = set(self._failed_snapshot)
        return out

    def get_finished(self):
        self._poll_copies()
        with self._mu:
            self._recving_snapshot = set(self._recving)
            return self._handoff_sending_locked(), set(self._recving_snapshot)

    def clear_published(self) -> None:
        with self._mu:
            for req_id in self._sending_snapshot:
                req = self._reqs.get(req_id)
                if req is not None:
                    req.sending_published = True
                self._sending.discard(req_id)
            self._sending_snapshot.clear()
            self._recving.difference_update(self._recving_snapshot)
            self._recving_snapshot.clear()
            self._failed.difference_update(self._failed_snapshot)
            self._failed_snapshot.clear()

    def _watermark_locked(self, blocks):
        """Bytes this snapshot will allocate, and the largest of those allocations."""
        if self.geometry is None:
            return 0, 0
        groups = {}
        tails = []
        for req_id, generation, block in blocks:
            if int(block.valid_tokens) >= int(self.geometry.block_size):
                groups.setdefault((str(req_id), int(generation)), []).append(block)
            else:
                tails.append(block)
        need = 0
        span = 0
        full = self.geometry.kvb_bytes(self.geometry.block_size)
        for grouped in groups.values():
            nbytes = full * len(grouped)
            need += nbytes
            if nbytes > span:
                span = nbytes
        for block in tails:
            nbytes = self.geometry.kvb_bytes(block.valid_tokens)
            need += nbytes
            if nbytes > span:
                span = nbytes
        return need, span

    def _backoff(self, table: dict, key: str) -> float:
        step = min(int(table.get(key, 0)) + 1, 8)
        table[key] = step
        return 0.02 * (2 ** (step - 1))

    def _track_open_locked(self, req: _Req, block: _Block) -> None:
        """One open entry per block id. Terminal blocks stay out of the list."""
        if block.saved or block.inflight or block.d2h_done:
            self._untrack_open_locked(req, block.block_id)
            return
        bid = int(block.block_id)
        for index, existing in enumerate(req.open_blocks):
            if int(existing.block_id) == bid:
                req.open_blocks[index] = block
                return
        req.open_blocks.append(block)

    def _untrack_open_locked(self, req: _Req, block_id: int) -> None:
        bid = int(block_id)
        req.open_blocks = [block for block in req.open_blocks if int(block.block_id) != bid]

    def _release_inflight_locked(self, req_id, generation, block: _Block) -> None:
        """The copy did not start. The next wait_for_save has to see this block."""
        block.inflight = False
        if block.saved or block.d2h_done:
            return
        req = self._reqs.get(str(req_id))
        if req is None or req.closed or int(req.generation) != int(generation):
            return
        self._track_open_locked(req, block)
        self._sync_save_active_locked(req)

    def _refresh_open_src_locked(self, req: _Req, block: _Block) -> None:
        if block.saved or block.inflight or block.d2h_done:
            return
        bases, events, epoch = self._layer_src_locked(req)
        if int(block.src_epoch) == int(epoch):
            return
        if self.geometry is None:
            return
        stride = int(block.stride) or int(self.geometry.layer_stride())
        block.stride = stride
        src = block.src
        src.clear()
        bid = int(block.block_id)
        for name, base in bases.items():
            src[name] = int(base) + bid * stride
            if name not in block.events:
                block.events[name] = events.get(name)
        block.src_epoch = int(epoch)

    def _refresh_open_events_locked(self, req: _Req, block: _Block) -> None:
        """Write this step's events into the existing dict. Do not allocate one."""
        if block.saved or block.inflight or block.d2h_done:
            return
        _bases, events, _epoch = self._layer_src_locked(req)
        block_events = block.events
        for name, ev in events.items():
            if ev is not None:
                block_events[name] = ev

    def _seal_open_locked(self, req: _Req, block: _Block) -> None:
        block.saved = True
        block.d2h_done = True
        self._release_saved_locked(req)

    def _keep_open_locked(self, req: _Req, block: _Block, blocks: list) -> bool:
        """Caller holds ``_mu``. Return true when the block still waits for more tokens."""
        if block.saved or block.inflight or block.d2h_done:
            return False
        self._refresh_open_src_locked(req, block)
        self._refresh_open_events_locked(req, block)
        if self.save_mode == "end" and self._save_event is not None:
            block.save_event = self._save_event
        if self._hit_covers_locked(req, block):
            self._seal_open_locked(req, block)
            return False
        full = self.geometry is not None and block.valid_tokens >= self.geometry.block_size
        if not full and not req.finished:
            return True
        if req.finished or _producer_recorded(block):
            block.inflight = True
            blocks.append((req.req_id, req.generation, block))
            return False
        return True

    def _clip_open_tail_locked(self, req: _Req, ids, tokens: int, block_size: int) -> None:
        if tokens <= 0 or not ids:
            return
        index = (int(tokens) - 1) // int(block_size)
        if index < 0 or index >= len(ids):
            return
        current = req.blocks.get(int(ids[index]))
        if current is None or current.saved or current.inflight or current.d2h_done:
            return
        start = index * int(block_size)
        valid = min(int(block_size), int(tokens) - start)
        if valid <= 0:
            return
        current.valid_tokens = valid
        current.token_index = index

    def _materialize_locked(self, req: _Req, block_ids) -> None:
        """Build only the tokens past ``built_tokens``. The open tail keeps its ``src`` dict."""
        try:
            if self.geometry is None:
                return
            ids = req.block_ids if block_ids is None else block_ids
            bases, events, epoch = self._layer_src_locked(req)
            if not ids or not bases:
                return
            block_size = int(self.geometry.block_size)
            if block_size <= 0:
                return
            tokens = int(req.num_tokens or len(req.tokens) or 0)
            limit = min(tokens, len(ids) * block_size)
            cursor = int(req.built_tokens)
            if cursor >= limit:
                self._clip_open_tail_locked(req, ids, tokens, block_size)
                return
            stride = int(self.geometry.layer_stride())
            while cursor < limit:
                index = cursor // block_size
                if index >= len(ids):
                    break
                start = index * block_size
                valid = min(block_size, limit - start)
                if valid <= 0:
                    break
                block_id = int(ids[index])
                current = req.blocks.get(block_id)
                if current is None:
                    src = {}
                    block_events = {}
                    for name, base in bases.items():
                        src[name] = int(base) + block_id * stride
                        block_events[name] = events.get(name)
                    current = _Block(
                        block_id,
                        valid,
                        src,
                        block_events,
                        token_index=index,
                        stride=stride,
                        src_epoch=int(epoch),
                    )
                    req.blocks[block_id] = current
                    self._track_open_locked(req, current)
                elif current.saved or current.inflight or current.d2h_done:
                    pass
                else:
                    current.valid_tokens = valid
                    current.token_index = index
                    current.stride = stride
                new_cursor = start + valid
                if new_cursor <= cursor:
                    break
                cursor = new_cursor
            req.built_tokens = cursor
        finally:
            self._sync_save_active_locked(req)

    def _publish(self, req_id, generation, phase, hit, closed=0) -> None:
        if self.board is None:
            return
        slot = int(self._slots.get(str(req_id), -1))
        if slot < 0:
            return
        self.board.publish_at(slot, int(generation), int(phase), int(hit), int(closed))

    def _publish_recompute(self, req_id, generation) -> None:
        """Lookup finished and will not be loaded. Do not leave the board PENDING."""
        key = str(req_id)
        with self._mu:
            req = self._reqs.get(key)
            if req is not None and req.closed:
                return
        if self.board is None:
            return
        slot = int(self._slots.get(key, -1))
        if slot < 0:
            return
        view = self.board.get_at(slot)
        if (
            view is not None
            and int(view.phase) == PHASE_CLOSED
            and int(view.generation) >= int(generation)
        ):
            return
        self.board.publish_at(slot, int(generation), PHASE_MISS, 0, 0)

    def _retry_or_miss(self, req_id, generation, exc) -> None:
        """One queue retry, then the same PHASE_MISS an index miss publishes."""
        req_id = str(req_id)
        if exc is None:
            _LOG.warning("lookup returned no hit req=%s generation=%s", req_id, generation)
        else:
            _LOG.warning("lookup failed req=%s generation=%s", req_id, generation, exc_info=exc)
        if self.queue.release_lookup(req_id, generation, self._backoff(self._lookup_backoff, req_id)):
            return
        self._lookup_backoff.pop(req_id, None)
        if not self.queue.mark_fallback(req_id, generation):
            return
        with self._mu:
            req = self._reqs.get(req_id)
            if req is not None and not req.closed and int(req.generation) == int(generation):
                req.phase = PHASE_MISS
                req.hit_tokens = 0
        self._publish(req_id, generation, PHASE_MISS, 0)

    def _read_failed(self, req_id, generation) -> None:
        """Lookup hit, then the KV read failed. Same miss as an empty index."""
        req_id = str(req_id)
        self._preload_backoff.pop(req_id, None)
        past_alloc = False
        with self._mu:
            req = self._reqs.get(req_id)
            if req is not None and not req.closed and int(req.generation) == int(generation):
                past_alloc = req.phase in (PHASE_PRELOADED, PHASE_H2D, PHASE_HBM)
        if not self.queue.mark_fallback(req_id, generation):
            return
        with self._mu:
            req = self._reqs.get(req_id)
            if req is not None and not req.closed and int(req.generation) == int(generation):
                req.phase = PHASE_MISS
                req.hit_tokens = 0
                req._preload_slots = None
        self._publish(req_id, generation, PHASE_MISS, 0)
        # The scheduler may already have allocated on PRELOADED, so it will
        # not note_alloc again. Return the lookup place here.
        if past_alloc and self.queue.release_lookup_window(req_id):
            self._wake_lookup.set()
        _LOG.warning("preload read failed req=%s generation=%s", req_id, generation)

    def _publish_wait_expired(self, req_id: str, generation: int) -> bool:
        """True once this lookup has already retried for publish_wait_s."""
        if self.publish_wait_s <= 0:
            return True
        key = (str(req_id), int(generation))
        started = self._publish_wait_since.get(key)
        now = _mono()
        if started is None:
            self._publish_wait_since[key] = now
            return False
        return now - float(started) >= self.publish_wait_s

    def _clear_publish_wait(self, req_id: str, generation: int) -> None:
        self._publish_wait_since.pop((str(req_id), int(generation)), None)

    def phase_diag_snapshot(self) -> str:
        """One aggregate line. Callers print it on a timer, never per request."""
        queue = self.queue
        wait = claimed = init_n = reading = preloaded = h2d = reserved_n = 0
        reserved_bytes = 0
        waiting_slots = set(queue._wait)
        order_slots = set(queue._order)
        order_n = 0
        for entry in list(queue.items):
            if entry is None:
                continue
            status = int(entry.status)
            if status == int(STATUS_LOOKUP_PENDING):
                if entry.lookup_claimed:
                    claimed += 1
                elif entry.slot in waiting_slots:
                    wait += 1
                elif entry.slot in order_slots:
                    order_n += 1
            elif status == int(STATUS_INIT) and not entry.preload_claimed:
                init_n += 1
            elif status == int(STATUS_L1_READING):
                reading += 1
            elif status == int(STATUS_PRELOADED):
                preloaded += 1
            elif status == int(STATUS_H2D_IN_FLIGHT):
                h2d += 1
            if entry.reserved:
                reserved_n += 1
                reserved_bytes += self._slot_bytes(entry.reserved)
        pool = self.preload_pool
        dram_free = -1 if pool is None else int(pool.ledger_free())
        return (
            "SURDATA_PHASE wait=%d claimed=%d order=%d publish_wait=%d init=%d reading=%d "
            "preloaded=%d h2d=%d dram_free=%d reserved=%d reserved_bytes=%d "
            "in_lookup=%d busy=%d skip_empty=%d skip_geometry=%d skip_full=%d"
            % (
                wait,
                claimed,
                order_n,
                len(self._publish_wait_since),
                init_n,
                reading,
                preloaded,
                h2d,
                dram_free,
                reserved_n,
                reserved_bytes,
                1 if self._lookup_inflight else 0,
                1 if self._lookup_busy else 0,
                int(self.h2d_skip_empty),
                int(self.h2d_skip_geometry),
                int(self.h2d_skip_full),
            )
        )

    def _phase_diag_loop(self) -> None:
        interval = max(0.001, float(self._phase_diag_ms) / 1000.0)
        while not self._stop.wait(interval):
            try:
                line = self.phase_diag_snapshot()
            except Exception as exc:
                _LOG.warning("SURDATA_PHASE snapshot failed: %s", exc)
                continue
            _LOG.warning("%s", line)

    def _lookup_loop(self) -> None:
        while not self._stop.is_set():
            item = self.queue.take_next_lookup()
            if item is None:
                self._wake_lookup.wait(self.queue.next_lookup_delay())
                self._wake_lookup.clear()
                continue
            req_id, generation, payload = item
            self._lookup_busy = 1
            try:
                self._lookup_one(req_id, generation, payload)
            finally:
                self._lookup_busy = 0
                self._lookup_inflight = 0

    def _lookup_one(self, req_id, generation, payload) -> None:
        tokens = list(payload.get("tokens") or [])
        reserved = self._reserve_lookup_dram(req_id, generation)
        if reserved is False:
            return
        block = 1 if self.geometry is None else self.geometry.block_size
        self._lookup_inflight = 1
        try:
            raw = self.sdk.lookup_hit(req_id, tokens, block, generation)
        except Exception as exc:
            self._lookup_inflight = 0
            self._abort_reserved(req_id)
            self._retry_or_miss(req_id, generation, exc)
            return
        self._lookup_inflight = 0
        if raw is None:
            self._abort_reserved(req_id)
            self._retry_or_miss(req_id, generation, None)
            return
        # logical_len may end inside a block. Flooring here turns a short tail into a miss.
        hit = max(0, int(raw))
        req_id = str(req_id)
        if hit <= 0:
            may = getattr(self.sdk, "prefix_may_arrive", None)
            waiting = callable(may) and may(req_id, tokens)
            expired = waiting and self._publish_wait_expired(req_id, generation)
            if waiting and not expired:
                with self._mu:
                    req = self._reqs.get(req_id)
                    if req is not None and not req.closed and int(req.generation) == int(generation):
                        req.phase = PHASE_PENDING
                        req.hit_tokens = 0
                if self.queue.requeue_lookup(req_id, generation, 0.05):
                    return
            if expired:
                started = self._publish_wait_since.get((req_id, int(generation)))
                waited = 0.0 if started is None else _mono() - float(started)
                _LOG.info(
                    "publish wait expired req=%s generation=%s waited_s=%.3f",
                    req_id,
                    generation,
                    waited,
                )
        self._clear_publish_wait(req_id, generation)
        if not self.queue.complete_lookup(req_id, generation, hit, len(tokens)):
            self.stale_drops += 1
            self._abort_reserved(req_id)
            drop = getattr(self.sdk, "drop_lookup", None)
            if callable(drop):
                drop(req_id, generation)
            self._publish_recompute(req_id, generation)
            return
        self._lookup_backoff.pop(req_id, None)
        phase = None
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or req.generation != generation:
                self.stale_drops += 1
            else:
                req.hit_tokens = hit
                if hit <= 0:
                    req.phase = PHASE_MISS
                else:
                    req.phase = PHASE_INIT
                phase = req.phase
        if phase is None:
            self._abort_reserved(req_id)
            self._publish_recompute(req_id, generation)
            return
        self._publish(req_id, generation, phase, hit)
        if hit <= 0:
            self._abort_reserved(req_id)
            return
        note = getattr(self.sdk, "note_hit_blocks", None)
        if callable(note):
            block = 1 if self.geometry is None else max(1, int(self.geometry.block_size))
            blocks = int(hit) // block
            if int(hit) % block:
                blocks += 1
            note(req_id, generation, blocks)
        if self.queue.claim_preload(req_id):
            self._start_one_preload(req_id, generation, hit)
        else:
            self._wake_preload.set()

    def _reserve_lookup_dram(self, req_id: str, generation: int):
        """Hold preload bytes for this prompt before the lookup. False means try later."""
        entry = self.queue.get(req_id)
        pool = self.preload_pool
        if entry is None or pool is None or self.geometry is None:
            return None
        if entry.reserved:
            return entry.reserved
        need = self._queue_need(entry)
        if need <= 0:
            return None
        capacity = int(getattr(pool, "slot_count", 0)) * int(getattr(pool, "slot_bytes", 0))
        if capacity and need > capacity:
            return None
        slots = pool.try_allocate(need, "preload", req_id, generation)
        if not slots:
            self.queue.defer_for_dram(req_id)
            return False
        self.queue.note_reserved(req_id, slots)
        return slots

    def _abort_reserved(self, req_id: str) -> None:
        slots = self.queue.take_reserved(req_id)
        if slots and self.preload_pool is not None:
            self.preload_pool.abort_unstored(slots)

    def _preload_loop(self) -> None:
        while not self._stop.is_set():
            item = self.queue.take_next_preload()
            if item is None:
                self._wake_preload.wait(self.queue.next_preload_delay())
                self._wake_preload.clear()
                continue
            req_id, generation, hit = item
            self._start_one_preload(req_id, generation, hit)

    def _start_one_preload(self, req_id, generation, hit) -> None:
        """Read the hit into the DRAM reserved for this lookup."""
        pool = self.preload_pool
        if self.geometry is None or pool is None:
            self._abort_reserved(req_id)
            self.queue.retry_preload(req_id, generation, 0.05)
            return
        block_size = int(self.geometry.block_size)
        blocks = max(1, (int(hit) + block_size - 1) // block_size)
        need = self.geometry.kvb_bytes(self.geometry.block_size) * blocks
        slots = self.queue.take_reserved(req_id)
        if slots and self._slot_bytes(slots) < need:
            pool.abort_unstored(slots)
            slots = None
        if not slots:
            slots = pool.try_allocate(need, "preload", req_id, generation)
        if not slots:
            self.queue.unclaim_preload(req_id, generation)
            return
        if not self.queue.mark_reading(req_id, generation):
            pool.abort_unstored(slots)
            self.queue.unclaim_preload(req_id, generation)
            return
        with self._mu:
            req = self._reqs.get(req_id)
            if req is not None and req.generation == generation and not req.closed:
                req.phase = PHASE_READING
        submit = getattr(self.sdk, "submit_read", None)
        if not callable(submit):
            pool.abort_unstored(slots)
            self._read_failed(req_id, generation)
            return
        try:
            rc = int(
                submit(
                    req_id,
                    generation,
                    hit,
                    slots,
                    lambda status, req_id=req_id, generation=generation, hit=hit, slots=slots: self._on_preloaded(
                        req_id, generation, hit, slots, status
                    ),
                )
            )
        except Exception:
            rc = -1
        if rc != 0:
            pool.abort_unstored(slots)
            self._read_failed(req_id, generation)

    def _slot_bytes(self, slots) -> int:
        total = 0
        for slot in slots or []:
            total += int(getattr(slot, "nbytes", 0))
        return total

    def _on_preloaded(self, req_id, generation, hit, slots, status) -> None:
        pool = self.preload_pool
        with self._mu:
            req = self._reqs.get(req_id)
            fresh = req is not None and not req.closed and req.generation == generation
        if not fresh or int(status) != 0:
            if pool is not None:
                pool.abort_unstored(slots)
            if fresh and int(status) != 0:
                self._read_failed(req_id, generation)
            else:
                self.stale_drops += 1
            return
        if not self.queue.mark_preloaded(req_id, generation, hit):
            if pool is not None:
                pool.unpin(slots)
            self.stale_drops += 1
            return
        if pool is not None:
            pool.mark_preloaded(slots)
        self._preload_backoff.pop(req_id, None)
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or req.generation != generation:
                if pool is not None:
                    pool.unpin(slots)
                self.stale_drops += 1
                return
            req.phase = PHASE_PRELOADED
            req.hit_tokens = int(hit)
            req._preload_slots = slots
        self._publish(req_id, generation, PHASE_PRELOADED, int(hit))
        if self.queue.release_lookup_window(req_id):
            self._wake_lookup.set()
        self._wake_h2d.set()

    def _fail_h2d(self, req_id: str, generation: int, unpin: bool) -> None:
        slots = None
        live = False
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is not None and not req.closed and int(req.generation) == int(generation):
                live = True
                if req.phase != PHASE_MISS:
                    req.phase = PHASE_MISS
                    self._failed.add(str(req_id))
                if unpin:
                    slots = getattr(req, "_preload_slots", None)
                    req._preload_slots = None
        if slots is not None and self.preload_pool is not None:
            self.preload_pool.unpin(slots)
        self.queue.mark_fallback(str(req_id), generation)
        self.queue.release_lookup_window(str(req_id))
        if not live:
            return
        self._publish(str(req_id), generation, PHASE_MISS, 0)
        self._wake_lookup.set()

    def _skip_unrequested_h2d(self, req_id: str, generation: int) -> None:
        """Drop a preload the scheduler will not wait for.

        ``external_tokens`` is 0 and block ids are already installed, so this
        step put the request on ``RUNNING``. Do not copy, and do not report
        the id in ``finished_recving`` or ``failed_recving``.
        """
        slots = None
        live = False
        with self._mu:
            req = self._reqs.get(str(req_id))
            if req is not None and not req.closed and int(req.generation) == int(generation):
                live = True
                if req.phase != PHASE_MISS:
                    req.phase = PHASE_MISS
                slots = getattr(req, "_preload_slots", None)
                req._preload_slots = None
        if slots is not None and self.preload_pool is not None:
            self.preload_pool.unpin(slots)
        self.queue.mark_fallback(str(req_id), generation)
        self.queue.release_lookup_window(str(req_id))
        if not live:
            return
        self._publish(str(req_id), generation, PHASE_MISS, 0)
        self._wake_lookup.set()

    def _submit_preloaded_h2d(self, req_id: str, generation: int) -> None:
        """Submit this prefix in one async H2D. Spans exist only for that call."""
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or int(req.generation) != int(generation):
                return
            if req.phase != PHASE_PRELOADED:
                return
            slots = getattr(req, "_preload_slots", None)
            block_ids = list(req.block_ids)
            hit_tokens = int(req.hit_tokens)
            external_tokens = int(req.external_tokens)
            full_table = bool(req.full_block_table)
            if not slots or not block_ids:
                self.h2d_skip_empty += 1
                return
            layer_bases = self._copy_layer_bases_locked(req)
            if self.geometry is None or len(layer_bases) < self.geometry.num_layers:
                self.h2d_skip_geometry += 1
                return
        block_size = int(self.geometry.block_size)
        full_count = int(hit_tokens) // block_size
        tail_tokens = int(hit_tokens) % block_size
        hit_blocks = full_count + (1 if tail_tokens else 0)
        if hit_blocks <= 0:
            return
        # A full scheduler table shorter than the hit will not grow. The
        # scheduler already counted the whole hit as external, so report a
        # miss instead of leaving the request in PRELOADED. An alloc list
        # that is not yet the scheduler table still waits for ingest_blocks.
        if len(block_ids) < hit_blocks:
            if not full_table:
                return
            _LOG.warning(
                "short full block table req=%s generation=%s blocks=%s hit_blocks=%s hit_tokens=%s",
                req_id,
                int(generation),
                len(block_ids),
                hit_blocks,
                int(hit_tokens),
            )
            self._fail_h2d(req_id, generation, unpin=True)
            return
        # Zero external tokens means this schedule is not waiting for a remote
        # load. An empty block list has already returned above and stays
        # PRELOADED until the scheduler allocates. A non-empty list is the
        # RUNNING admission: release the preload and do not copy.
        if int(external_tokens) <= 0:
            if block_ids:
                self._skip_unrequested_h2d(req_id, generation)
            return
        # An alloc list can start after a local GPU prefix. Slice it only when
        # this step's external tokens already cover the whole SurData hit, or
        # after ingest_blocks has stored the scheduler table from token 0.
        if not full_table and int(external_tokens) < int(hit_tokens):
            return
        prefix = block_ids[:hit_blocks]
        stride = self.geometry.layer_stride()
        payload = self.geometry.layer_payload(self.geometry.block_size)
        names = list(layer_bases)[: self.geometry.num_layers]
        count = hit_blocks
        try:
            spans = []
            run_start = 0
            full_prefix = prefix[:full_count]
            for bi, block_id in enumerate(full_prefix):
                nxt = bi + 1 == len(full_prefix) or int(full_prefix[bi + 1]) != int(block_id) + 1
                if not nxt:
                    continue
                run = bi + 1 - run_start
                for li, name in enumerate(names):
                    spans.append(
                        {
                            "src": slot_addr(slots, (li * count + run_start) * payload),
                            "dst": int(layer_bases[name]) + int(full_prefix[run_start]) * stride,
                            "bytes": payload * run,
                            "layer": li,
                        }
                    )
                run_start = bi + 1
            if tail_tokens:
                short = self.geometry.layer_payload(tail_tokens)
                for li, name in enumerate(names):
                    spans.append(
                        {
                            "src": slot_addr(slots, (li * count + full_count) * payload),
                            "dst": int(layer_bases[name]) + int(prefix[full_count]) * stride,
                            "bytes": short,
                            "layer": li,
                        }
                    )
        except Exception:
            self._fail_h2d(req_id, generation, unpin=True)
            return
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or int(req.generation) != int(generation):
                return
            if req.phase != PHASE_PRELOADED:
                return
            req.phase = PHASE_H2D
            try:
                self._publish(req_id, generation, PHASE_H2D, 0)
            except Exception:
                try:
                    self._publish(req_id, generation, PHASE_H2D, 0)
                except Exception:
                    pass
        job = self._job(req_id, generation)
        self._jobs[job["sequence"]] = (CopyKind.H2D, req_id, generation, slots, job)
        try:
            rc = int(self.native.submit_h2d(job, spans, generation, generation) or 0)
        except Exception:
            self._jobs.pop(job["sequence"], None)
            self._fail_h2d(req_id, generation, unpin=True)
            return
        if rc == 1:
            self._jobs.pop(job["sequence"], None)
            self.h2d_skip_full += 1
            with self._mu:
                req = self._reqs.get(req_id)
                if req is not None and not req.closed and int(req.generation) == int(generation):
                    req.phase = PHASE_PRELOADED
                    self._publish(req_id, generation, PHASE_PRELOADED, hit_tokens)
            return
        if rc != 0:
            self._jobs.pop(job["sequence"], None)
            self._fail_h2d(req_id, generation, unpin=True)
            return
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or int(req.generation) != int(generation):
                req = None
        if req is None:
            self._cancel_native_job(job)
            return
        self.queue.mark_h2d(req_id, generation)

    def _drain_parks(self) -> None:
        for key, generation in list(self._park_wait.items()):
            self._submit_park(key, int(generation))

    def _park_spans(self, slots, block_ids, layer_bases, payload: int):
        stride = self.geometry.layer_stride()
        names = list(layer_bases)
        spans = []
        count = len(block_ids)
        run_start = 0
        for bi, block_id in enumerate(block_ids):
            nxt = bi + 1 == count or int(block_ids[bi + 1]) != int(block_id) + 1
            if not nxt:
                continue
            run = bi + 1 - run_start
            for li, name in enumerate(names):
                spans.append(
                    {
                        "src": int(layer_bases[name]) + int(block_ids[run_start]) * stride,
                        "dst": slot_addr(slots, (li * count + run_start) * payload),
                        "bytes": payload * run,
                        "layer": li,
                    }
                )
            run_start = bi + 1
        return spans

    def _submit_park(self, req_id: str, generation: int) -> None:
        """Copy computed KV into a new preload span and leave it PRELOADED."""
        key = str(req_id)
        with self._mu:
            if int(self._park_wait.get(key, -1)) != int(generation):
                return
            if key in self._park_submitting:
                return
            req = self._reqs.get(key)
            if req is None or req.closed or req.finished or int(req.generation) != int(generation):
                self._park_wait.pop(key, None)
                return
            block_ids = [int(block) for block in req.block_ids]
            layer_bases = self._copy_layer_bases_locked(req)
            tokens = int(req.num_tokens or 0)
            self._park_submitting.add(key)
        try:
            if (
                self.geometry is None
                or self.preload_pool is None
                or not block_ids
                or len(layer_bases) < self.geometry.num_layers
            ):
                return
            count = len(block_ids)
            payload = self.geometry.layer_payload(self.geometry.block_size)
            if tokens <= 0:
                tokens = count * int(self.geometry.block_size)
            need = payload * count * len(layer_bases)
            slots = self.preload_pool.try_allocate(need, "preload", key, generation)
            if not slots:
                return
            spans = self._park_spans(slots, block_ids, layer_bases, payload)
            job = self._job(key, generation)
            self._jobs[job["sequence"]] = (CopyKind.PARK, key, int(generation), slots, int(tokens), job)
            try:
                rc = int(self.native.submit_d2h(job, spans, generation, generation, []) or 0)
            except Exception:
                self._jobs.pop(job["sequence"], None)
                self.preload_pool.abort_unstored(slots)
                return
            if rc != 0:
                self._jobs.pop(job["sequence"], None)
                self.preload_pool.abort_unstored(slots)
                return
            with self._mu:
                req = self._reqs.get(key)
                if req is None or req.closed or int(req.generation) != int(generation):
                    self._cancel_native_job(job)
                    return
                self._park_wait.pop(key, None)
        finally:
            with self._mu:
                self._park_submitting.discard(key)

    def _h2d_loop(self) -> None:
        # mark_preloaded 之后，lookup 和 preload 不再写这一行。
        # 只有本线程把它改成 H2D_IN_FLIGHT 或 FALLBACK_MISS，进入 HBM 后把这一行摘掉，所以这里不拿队列锁。
        while not self._stop.is_set():
            self._drain_parks()
            for slot in list(self.queue._order):
                entry = self.queue.items[slot] if 0 <= int(slot) < len(self.queue.items) else None
                if entry is None or entry.status != STATUS_PRELOADED or entry.actual_tokens <= 0:
                    continue
                key = entry.req_id
                with self._mu:
                    live = self._reqs.get(key)
                    already = (
                        live is not None
                        and not live.closed
                        and int(live.generation) == int(entry.generation)
                        and live.phase == PHASE_H2D
                    )
                if already:
                    self.queue.mark_h2d(key, int(entry.generation))
                    continue
                try:
                    self._submit_preloaded_h2d(key, int(entry.generation))
                except Exception:
                    self._fail_h2d(key, int(entry.generation), unpin=True)
            self._poll_copies()
            self._wake_h2d.wait(0.05)
            self._wake_h2d.clear()

    def _d2h_loop(self) -> None:
        while not self._stop.is_set():
            with self._mu:
                blocks = self._snaps.pop(0) if self._snaps else None
                if blocks:
                    for req_id, generation, block in blocks:
                        self._d2h_handed.add((str(req_id), int(generation), int(block.block_id)))
            if blocks is not None:
                grouped = self._d2h_groups(blocks)
                for req_id, generation, chunk in grouped:
                    try:
                        if len(chunk) > 1:
                            self._submit_d2h_group(req_id, generation, chunk)
                        else:
                            self._submit_one_d2h(req_id, generation, chunk[0])
                    finally:
                        with self._mu:
                            for block in chunk:
                                self._d2h_handed.discard(
                                    (str(req_id), int(generation), int(block.block_id))
                                )
                self._poll_copies()
                continue
            self._poll_copies()
            self._wake_d2h.wait(0.05)
            self._wake_d2h.clear()

    def note_compute_stream(self, stream=None) -> None:
        """Remember the worker compute stream. D2H waits for this stream."""
        if stream is not None:
            self._compute_stream = stream
            return
        torch = self._torch_module()
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is None or not cuda.is_available():
            return
        self._compute_stream = cuda.current_stream()

    def _record_tail_event(self):
        """Record one event on the compute stream. Does not synchronize."""
        torch = self._torch_module()
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is None or not cuda.is_available():
            return None
        stream = self._compute_stream or cuda.current_stream()
        event = cuda.Event(enable_timing=False)
        event.record(stream)
        return event

    def _torch_module(self):
        if self._torch is not None:
            return self._torch
        import torch

        return torch

    def _d2h_groups(self, blocks):
        """Full blocks of one request share one allocation. A short tail stays alone."""
        groups = []
        pending = {}
        for req_id, generation, block in blocks:
            full = (
                self.geometry is not None
                and int(block.valid_tokens) >= int(self.geometry.block_size)
            )
            key = (str(req_id), int(generation))
            if not full:
                groups.append((req_id, generation, [block]))
                continue
            chunk = pending.get(key)
            if chunk is None:
                chunk = []
                pending[key] = chunk
                groups.append((req_id, generation, chunk))
            chunk.append(block)
        return groups

    def _submit_d2h_group(self, req_id, generation, blocks) -> bool:
        if self.geometry is None or self.store_pool is None:
            with self._mu:
                for block in blocks:
                    self._release_inflight_locked(req_id, generation, block)
            return True
        with self._mu:
            req = self._reqs.get(req_id)
            finished = req is not None and req.finished and int(req.generation) == int(generation)
        if not finished:
            for block in blocks:
                if block.save_event is not None:
                    continue
                for layer in block.src:
                    if block.events.get(layer) is None:
                        with self._mu:
                            for one in blocks:
                                self._release_inflight_locked(req_id, generation, one)
                        return True
        need = self.geometry.kvb_bytes(self.geometry.block_size) * len(blocks)
        slots = self.store_pool.try_allocate(need, "store", req_id, generation)
        if not slots:
            for block in blocks:
                self._on_d2h_blocked(req_id, generation, block)
            return True
        for _extra in range(len(blocks) - 1):
            self.store_pool.pin(slots)
        group = {
            "slots": slots,
            "left": len(blocks),
            "failed": False,
            "count": len(blocks),
            "lock": threading.Lock(),
        }
        try:
            event, kept = self._copy_blocks(blocks, slots)
        except Exception as exc:
            print("SURDATA_D2H copy failed: %s" % exc, file=sys.stderr, flush=True)
            self.store_pool.abort_unstored(slots)
            for block in blocks:
                self._on_d2h_blocked(req_id, generation, block)
            return True
        with self._mu:
            self._d2h_inflight.append(
                {
                    "event": event,
                    "kept": kept,
                    "req_id": req_id,
                    "generation": generation,
                    "slots": slots,
                    "block": blocks[0],
                    "blocks": list(blocks),
                    "group": group,
                }
            )
        return True

    def _submit_one_d2h(self, req_id, generation, block: _Block) -> bool:
        if self.geometry is None or self.store_pool is None:
            with self._mu:
                self._release_inflight_locked(req_id, generation, block)
            return True
        with self._mu:
            req = self._reqs.get(req_id)
            finished = req is not None and req.finished and int(req.generation) == int(generation)
        if not finished and block.save_event is None:
            for layer in block.src:
                if block.events.get(layer) is None:
                    with self._mu:
                        self._release_inflight_locked(req_id, generation, block)
                    return True
        need = self.geometry.kvb_bytes(block.valid_tokens)
        slots = self.store_pool.try_allocate(need, "store", req_id, generation)
        if not slots:
            self._on_d2h_blocked(req_id, generation, block)
            return True
        try:
            event, kept = self._copy_block(block, slots)
        except Exception as exc:
            print("SURDATA_D2H copy failed: %s" % exc, file=sys.stderr, flush=True)
            self.store_pool.abort_unstored(slots)
            self._on_d2h_blocked(req_id, generation, block)
            return True
        with self._mu:
            self._d2h_inflight.append(
                {
                    "event": event,
                    "kept": kept,
                    "req_id": req_id,
                    "generation": generation,
                    "slots": slots,
                    "block": block,
                }
            )
        return True

    def _copy_block(self, block: _Block, slots):
        return self._copy_blocks([block], slots)

    def _copy_blocks(self, blocks, slots):
        """Wait for this forward, then one memcpy per contiguous layer run.

        ``per_layer`` waits for the whole compute stream. ``end`` waits for the
        single event recorded at the start of ``wait_for_save``.
        """
        torch = self._torch_module()
        cuda = torch.cuda
        if self._store_stream is None:
            self._store_stream = cuda.Stream()
        store = self._store_stream
        save_event = None
        for block in blocks:
            if block.save_event is not None:
                save_event = block.save_event
                break
        if save_event is not None:
            store.wait_event(save_event)
        else:
            compute = self._compute_stream or cuda.current_stream()
            store.wait_stream(compute)
        count = len(blocks)
        tokens = self.geometry.block_size if count > 1 else blocks[0].valid_tokens
        payload = self.geometry.layer_payload(tokens)
        memcpy = getattr(self.native, "memcpy_async", None)
        if not callable(memcpy):
            raise RuntimeError("memcpy_async missing")
        stream_ptr = store.cuda_stream
        names = list(blocks[0].src)
        spans = []
        run_start = 0
        for bi, block in enumerate(blocks):
            nxt = bi + 1 == count or int(blocks[bi + 1].block_id) != int(block.block_id) + 1
            if not nxt:
                continue
            run = bi + 1 - run_start
            for li, name in enumerate(names):
                src = int(blocks[run_start].src[name])
                dst = slot_addr(slots, (li * count + run_start) * payload)
                nbytes = int(payload) * run
                rc = int(memcpy(int(dst), int(src), nbytes, stream_ptr, 2))
                if rc != 0:
                    raise RuntimeError("memcpy_async rc=%s" % rc)
                spans.append({"src": int(src), "dst": int(dst), "bytes": nbytes, "layer": li})
            run_start = bi + 1
        event = cuda.Event(enable_timing=False)
        event.record(store)
        self.d2h_copies.append(spans)
        return event, None

    def _on_d2h_blocked(self, req_id, generation, block: _Block) -> None:
        """Pool full, or the copy failed after its slot was returned.

        ``drop`` abandons the block immediately. ``finish_drop`` does that only
        after the request has finished; a live request clears inflight so the
        next ``wait_for_save`` submits it again. Neither path queues a retry.
        """
        with self._mu:
            req = self._reqs.get(str(req_id))
            finished = (
                req is not None
                and int(req.generation) == int(generation)
                and req.finished
            )
        if finished or self.policy == "drop":
            self._drop_block(req_id, generation, block)
            return
        with self._mu:
            self._release_inflight_locked(req_id, generation, block)

    def _drop_block(self, req_id, generation, block: _Block) -> None:
        with self._mu:
            block.saved = True
            block.inflight = False
            block.d2h_done = True
            req = self._reqs.get(req_id)
            if req is not None and req.generation == generation:
                self._untrack_open_locked(req, block.block_id)
                self._forget_handed_locked(req, block.block_id)
                self._release_saved_locked(req)
            self._notes.append((str(req_id), int(generation), "skip", int(block.token_index)))
        self.drops.append((req_id, generation, block.block_id))
        self._wake_store.set()

    def _store_loop(self) -> None:
        while not self._stop.is_set():
            with self._mu:
                notes = self._notes
                self._notes = []
                item = self._store.pop(0) if self._store else None
                if item is not None:
                    self._storing = (str(item[0]), int(item[1]))
            ready = False
            if notes:
                apply = getattr(self.sdk, "apply_notes", None)
                if callable(apply):
                    ready = bool(apply(notes))
            if item is None:
                if ready:
                    self._wake_commit.set()
                else:
                    self._wake_store.wait(0.05)
                    self._wake_store.clear()
                continue
            req_id, generation, slots, block = item[:4]
            block_index = int(item[4]) if len(item) > 4 else 0
            block_count = int(item[5]) if len(item) > 5 else 1
            group = item[6] if len(item) > 6 else None
            wake_commit = ready
            try:
                if slots and all(getattr(slot, "stored", False) for slot in slots):
                    continue
                rc = self.sdk.enqueue_save(
                    req_id,
                    generation,
                    block.valid_tokens,
                    0 if self.geometry is None else self.geometry.block_size,
                    slots,
                    int(block.token_index),
                    lambda status, req_id=req_id, generation=generation, slots=slots, group=group: (
                        self._on_stored(req_id, generation, slots, status)
                        if group is None
                        else self._on_group_stored(group, status)
                    ),
                    block_index,
                    block_count,
                )
                if rc == -9:
                    with self._mu:
                        self._store.append(item)
                    self._wake_store.wait(0.05)
                    self._wake_store.clear()
                elif rc != 0 and group is not None:
                    self._on_group_stored(group, rc)
                elif rc != 0 and self.store_pool is not None:
                    self.store_pool.abort_unstored(slots)
                wake_commit = True
            finally:
                with self._mu:
                    self._storing = None
                    if item is not None:
                        req = self._reqs.get(str(item[0]))
                        if req is not None and int(req.generation) == int(item[1]):
                            self._release_saved_locked(req)
            if wake_commit:
                self._wake_commit.set()

    def _commit_loop(self) -> None:
        while not self._stop.is_set():
            self._wake_commit.wait(0.05)
            self._wake_commit.clear()
            self._commit_once()
        self._commit_once()

    def _commit_once(self) -> None:
        fn = getattr(self.sdk, "commit_ready", None)
        if callable(fn):
            fn(self._writes_settled)

    def _writes_settled(self, req_id, generation) -> bool:
        req_id = str(req_id)
        generation = int(generation)
        with self._mu:
            if self._storing == (req_id, generation):
                return False
            for item in self._store:
                if str(item[0]) == req_id and int(item[1]) == generation:
                    return False
            for item in self._d2h_inflight:
                if str(item.get("req_id")) == req_id and int(item.get("generation", -1)) == generation:
                    return False
            req = self._reqs.get(req_id)
            if req is None or int(req.generation) != generation:
                return True
            if not req.finished:
                return False
            return not self._d2h_busy_locked(req)

    def _on_stored(self, req_id, generation, slots, status) -> None:
        pool = self.store_pool
        with self._mu:
            req = self._reqs.get(req_id)
            fresh = req is not None and not req.closed and req.generation == generation
        if pool is None:
            return
        if not fresh or int(status) != 0:
            self.stale_drops += 1
            pool.abort_unstored(slots)
            return
        pool.mark_stored(slots)
        pool.unpin(slots)

    def _on_group_stored(self, group, status) -> None:
        """One save of a shared region finished. Free it after every block has returned."""
        with group["lock"]:
            if int(status) != 0:
                group["failed"] = True
            group["left"] -= 1
            if group["left"] > 0:
                return
            failed = bool(group["failed"])
            slots = group["slots"]
            count = int(group["count"])
        pool = self.store_pool
        if pool is None:
            return
        if failed:
            pool.abort_unstored(slots)
            return
        pool.mark_stored(slots)
        for _pin in range(count):
            pool.unpin(slots)

    def _gc_loop(self) -> None:
        while not self._stop.is_set():
            pool = self.store_pool
            if pool is None:
                self._stop.wait(0.05)
                continue
            pool.watermark_event().wait(0.05)
            pool.watermark_event().clear()
            if self._stop.is_set():
                return
            pool.reclaim_to_watermark()

    def _poll_copies(self) -> None:
        self._poll_torch_d2h()
        done = self.native.poll(32)
        for item in done:
            sequence = int(item.get("sequence", 0))
            kind = self._jobs.pop(sequence, None)
            if kind is None:
                continue
            direction = kind[0]
            if direction == CopyKind.H2D:
                _, req_id, generation, slots = kind[:4]
                ok = int(item.get("status", 0)) == 0
                if slots and self.preload_pool is not None:
                    self.preload_pool.unpin(slots)
                opened = False
                miss = False
                with self._mu:
                    req = self._reqs.get(req_id)
                    if req is None or req.closed or req.generation != generation:
                        self.stale_drops += 1
                        continue
                    if req.phase == PHASE_MISS:
                        continue
                    if ok:
                        req.phase = PHASE_HBM
                        req._preload_slots = None
                        self._recving.add(req_id)
                        opened = self.queue.drop_loaded(req_id, generation)
                        self._publish(req_id, generation, PHASE_HBM, 0)
                    else:
                        miss = True
                if miss:
                    self._fail_h2d(req_id, generation, unpin=False)
                if opened:
                    self._wake_lookup.set()
            elif direction == CopyKind.PARK:
                _, req_id, generation, slots, tokens = kind[:5]
                ok = int(item.get("status", 0)) == 0
                if not ok:
                    if slots and self.preload_pool is not None:
                        self.preload_pool.abort_unstored(slots)
                    with self._mu:
                        req = self._reqs.get(req_id)
                        if (
                            req is not None
                            and not req.closed
                            and not req.finished
                            and int(req.generation) == int(generation)
                        ):
                            self._park_wait[str(req_id)] = int(generation)
                    continue
                finished = False
                board_slot = None
                with self._mu:
                    req = self._reqs.get(req_id)
                    if req is None or req.closed or int(req.generation) != int(generation):
                        self.stale_drops += 1
                        req = None
                    elif req.finished:
                        finished = True
                    else:
                        req.phase = PHASE_PRELOADED
                        req.hit_tokens = int(tokens)
                        req.block_ids = []
                        req._preload_slots = slots
                        board_slot = int(self._slots.get(str(req_id), req.slot))
                if req is None:
                    if slots and self.preload_pool is not None:
                        self.preload_pool.abort_unstored(slots)
                    continue
                if finished:
                    self.queue.drop_loaded(req_id, generation)
                    self._move_park_bytes_to_store(req_id, generation, slots, int(tokens))
                    continue
                if self.preload_pool is not None:
                    self.preload_pool.mark_preloaded(slots)
                self.queue.insert_preloaded(
                    req_id, generation, int(tokens), slot=None if board_slot is None or board_slot < 0 else board_slot
                )
                self._publish(req_id, generation, PHASE_PRELOADED, int(tokens))
                self._wake_h2d.set()
            elif direction == CopyKind.D2H and len(kind) >= 5 and isinstance(kind[4], int):
                self._finish_store_copy(kind, item)
            elif direction == CopyKind.D2H:
                _, req_id, generation, slots, block = kind
                if int(item.get("status", 0)) != 0:
                    if self.store_pool is not None:
                        self.store_pool.abort_unstored(slots)
                    self._on_d2h_blocked(req_id, generation, block)
                    continue
                store_it = False
                with self._mu:
                    req = self._reqs.get(req_id)
                    if req is None or req.closed or req.generation != generation:
                        self.stale_drops += 1
                        if self.store_pool is not None:
                            self.store_pool.abort_unstored(slots)
                        continue
                    block.d2h_done = True
                    block.inflight = False
                    self._untrack_open_locked(req, block.block_id)
                    self._forget_handed_locked(req, block.block_id)
                    if self._block_past_tokens(req, block):
                        self._release_saved_locked(req)
                    else:
                        self._store.append((req_id, generation, slots, block))
                        store_it = True
                        self._release_saved_locked(req)
                if store_it:
                    self._wake_store.set()
                elif self.store_pool is not None:
                    self.store_pool.abort_unstored(slots)

    def _poll_torch_d2h(self) -> None:
        with self._mu:
            pending = list(self._d2h_inflight)
            self._d2h_inflight = []
            polling = []
            for item in pending:
                owned = item.get("blocks") or ([item["block"]] if item.get("block") is not None else [])
                for block in owned:
                    key = (str(item.get("req_id")), int(item.get("generation", -1)), int(block.block_id))
                    self._d2h_handed.add(key)
                    polling.append(key)
        still = []
        try:
            for item in pending:
                try:
                    ready = bool(item["event"].query())
                except Exception:
                    ready = False
                    item["status"] = 3
                    self._apply_d2h_result(item)
                    continue
                if not ready:
                    still.append(item)
                    continue
                item["status"] = 0
                self._apply_d2h_result(item)
        finally:
            with self._mu:
                if still:
                    self._d2h_inflight = still + self._d2h_inflight
                for key in polling:
                    self._d2h_handed.discard(key)

    def _apply_d2h_result(self, item) -> None:
        req_id = item["req_id"]
        generation = item["generation"]
        slots = item["slots"]
        owned = item.get("blocks") or [item["block"]]
        group = item.get("group")
        item["kept"] = None
        if int(item.get("status", 0)) != 0:
            if self.store_pool is not None:
                self.store_pool.abort_unstored(slots)
            for block in owned:
                self._on_d2h_blocked(req_id, generation, block)
            return
        stored = 0
        skipped = 0
        with self._mu:
            req = self._reqs.get(req_id)
            if req is None or req.closed or req.generation != generation:
                self.stale_drops += 1
                if self.store_pool is not None:
                    self.store_pool.abort_unstored(slots)
                return
            for index, block in enumerate(owned):
                block.d2h_done = True
                block.inflight = False
                self._untrack_open_locked(req, block.block_id)
                self._forget_handed_locked(req, block.block_id)
                if self._block_past_tokens(req, block):
                    skipped += 1
                    continue
                stored += 1
                self._store.append(
                    (req_id, generation, slots, block, index, len(owned), group)
                )
            self._release_saved_locked(req)
        if stored == 0:
            if self.store_pool is not None:
                self.store_pool.abort_unstored(slots)
            return
        if group is not None:
            for _ignored in range(skipped):
                self._on_group_stored(group, 0)
        self._wake_store.set()

    def _job(self, req_id, generation) -> dict:
        with self._mu:
            sequence = self._seq
            self._seq += 1
        return {
            "engine_epoch": 1,
            "worker_epoch": 1,
            "sequence": sequence,
            "req": req_id,
            "generation": generation,
        }


def _byte_view(torch_mod, ptr: int, nbytes: int, device: str):
    """Non-owning uint8 view. Same helper LMCache uses for a raw pointer.

    torch 2.14 keeps an unindexed ``cuda`` device on the storage, then
    ``empty(..., device="cuda")`` lands on ``cuda:0`` and ``set_`` refuses.
    Index the CUDA device before building the storage, and build the empty
    tensor on ``storage.device``.
    """
    hook = getattr(torch_mod, "byte_view", None)
    if callable(hook):
        return hook(int(ptr), int(nbytes), device)
    dev = torch_mod.device(device)
    if getattr(dev, "type", None) == "cuda" and getattr(dev, "index", None) is None:
        cuda = getattr(torch_mod, "cuda", None)
        if cuda is not None:
            dev = torch_mod.device("cuda", int(cuda.current_device()))
    storage = torch_mod._C._construct_storage_from_data_pointer(int(ptr), dev, int(nbytes))
    tensor = torch_mod.empty(0, dtype=torch_mod.uint8, device=getattr(storage, "device", dev))
    tensor.set_(storage, 0, (int(nbytes),), (1,))
    return tensor


def _buf_addr(buf, offset=0) -> int:
    import ctypes

    if isinstance(buf, int):
        return int(buf) + int(offset)
    return ctypes.addressof(ctypes.c_char.from_buffer(buf)) + int(offset)


def slot_addr(slots, offset) -> int:
    left = int(offset)
    for slot in slots:
        if left < slot.nbytes:
            return _buf_addr(slot.buf, left)
        left -= slot.nbytes
    raise RuntimeError("KVB offset is outside the pinned slots")


def _calls(obj, tid) -> int:
    table = getattr(obj, "call_threads", None)
    if table is None:
        return 0
    try:
        items = tuple(table)
    except RuntimeError:
        return 0
    return sum(1 for item in items if item == tid)
