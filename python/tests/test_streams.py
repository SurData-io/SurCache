import ctypes
import errno
import inspect
import io
import mmap
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from surdata_vllm.config import (
    ConfigError,
    load_direct_phase_diag_ms,
    load_direct_save_mode,
    load_settings,
    load_store_full_policy,
)
from surdata_vllm.kvb import KvGeometry
from surdata_vllm.layout import LayoutError, as_mha_axes, classify_kv_shape
from surdata_vllm.sdk_bridge import abi_sizes
from surdata_vllm.store_pool import StorePool


def _transfer_so():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib-so", "libsurdata_transfer.so")
    )


_HOSTMEM = None


def _hostmem_lib():
    global _HOSTMEM
    if _HOSTMEM is None:
        lib = ctypes.CDLL(_transfer_so())
        lib.surdata_hostmem_alloc.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.surdata_hostmem_alloc.restype = ctypes.c_int
        lib.surdata_hostmem_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.surdata_hostmem_free.restype = None
        _HOSTMEM = lib
    return _HOSTMEM


def alloc_hostmem(nbytes):
    out = ctypes.c_void_p()
    mapped = ctypes.c_size_t()
    rc = int(
        _hostmem_lib().surdata_hostmem_alloc(
            ctypes.c_size_t(int(nbytes)), ctypes.byref(out), ctypes.byref(mapped)
        )
    )
    if rc != 0 or not out.value:
        raise OSError(rc, os.strerror(rc) if 0 < rc < 256 else "surdata_hostmem_alloc")
    return int(out.value), int(mapped.value)


def free_hostmem(ptr, mapped):
    if not ptr:
        return
    _hostmem_lib().surdata_hostmem_free(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(mapped)))


def locked_pool(slot_bytes, slot_count=None, capacity_bytes=None, **kwargs):
    if capacity_bytes is None:
        if slot_count is None:
            raise ValueError("slot_count or capacity_bytes is required")
        capacity_bytes = int(slot_bytes) * int(slot_count)
    base, mapped = alloc_hostmem(int(capacity_bytes))
    try:
        return StorePool(
            slot_bytes,
            slot_count,
            capacity_bytes=int(capacity_bytes),
            base_ptr=base,
            mapped_bytes=mapped,
            release=free_hostmem,
            **kwargs,
        )
    except Exception:
        free_hostmem(base, mapped)
        raise


def _pin_text():
    with open("/tmp/surdata_pin.txt", encoding="utf-8") as handle:
        return handle.read()


def _vm_lck_kb():
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmLck:"):
                return int(line.split()[1])
    return 0
from surdata_vllm.dram_queue import (
    STATUS_DONE,
    STATUS_FALLBACK_MISS,
    STATUS_H2D_IN_FLIGHT,
    STATUS_LOOKUP_PENDING,
    STATUS_PRELOADED,
    DramPipelineQueue,
)
from surdata_vllm.streams import (
    CopyKind,
    PHASE_CLOSED,
    PHASE_H2D,
    PHASE_HBM,
    PHASE_INIT,
    PHASE_MISS,
    PHASE_PENDING,
    PHASE_PRELOADED,
    PHASE_READING,
    SchedulerFront,
    StatusBoard,
    WorkerRuntime,
    _Req,
    _SLOTS,
    _byte_view,
    _process_boot_id,
    open_scheduler_board,
    open_worker_board,
    slot_addr,
    status_board_path,
)


def plant_board(path, boot, slot, generation, phase, hit):
    size = 16 + 1024 * 128
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size)
        try:
            struct.pack_into("<IIII", mm, 0, 0x53564431, 1024, int(boot), 0)
            off = 16 + int(slot) * 128
            struct.pack_into("<IIIII", mm, off, 2, int(generation), int(phase), int(hit), 0)
        finally:
            mm.close()
    finally:
        os.close(fd)


def read_boot(path):
    with open(path, "rb") as handle:
        raw = handle.read(12)
    return struct.unpack_from("<I", raw, 8)[0]


def wait_until(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class FakeSdk:
    def __init__(self):
        self.scripted = {}
        self.gates = {}
        self.entered = []
        self.reads = []
        self.fail_lookup = False
        self.fail_read = False
        self.saves = []
        self.save_slots = []
        self.tails = []
        self.visible = 0
        self.busy = False
        self.hold_save = False
        self.pending = []
        self.fail_read = False
        self.read_status = 0
        self.read_gate = None
        self.call_threads = []
        self.geometry = None
        self._lock = threading.Lock()

    def lookup_hit(self, req_id, tokens, block_tokens, generation):
        del req_id
        if self.fail_lookup:
            raise RuntimeError("lookup busy")
        with self._lock:
            self.entered.append(int(generation))
            self.call_threads.append(threading.get_ident())
            gate = self.gates.get(int(generation))
        if gate is not None:
            gate.wait(2)
        if int(generation) in self.scripted:
            return int(self.scripted[int(generation)])
        full = (len(tokens) // max(1, int(block_tokens))) * int(block_tokens)
        return min(self.visible, full)

    def drop_lookup(self, req_id, generation):
        del req_id, generation

    def submit_read(self, req_id, generation, hit, slots, on_done):
        del slots
        with self._lock:
            self.reads.append((req_id, generation, hit))
            self.call_threads.append(threading.get_ident())
        if self.fail_read:
            return -1
        gate = self.read_gate
        if gate is not None:
            def waiter():
                gate.wait(3)
                on_done(int(self.read_status))

            threading.Thread(target=waiter, daemon=True).start()
            return 0
        on_done(int(self.read_status))
        return 0

    def enqueue_save(
        self, req_id, generation, valid_tokens, block_tokens, slots, token_index, on_done, block_index=0, block_count=1
    ):
        with self._lock:
            self.call_threads.append(threading.get_ident())
            if self.busy:
                return -9
            if self.hold_save:
                self.pending.append(on_done)
                return 0
            if int(valid_tokens) >= int(block_tokens):
                self.visible += int(block_tokens)
            else:
                self.tails.append(int(valid_tokens))
            self.saves.append((req_id, generation, int(valid_tokens), int(token_index)))
            self.save_slots.append(slots)
        on_done(0)
        return 0


class FakeNative:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = {}
        self.next_ev = 1
        self.waiting = []
        self.done = []
        self.fail_h2d = False
        self.h2d_rcs = []
        self.h2d_spans = []
        self.h2d_hook = None
        self.syncs = 0
        self.call_threads = []
        self.submits = []
        self.spans = []
        self.hold = False
        self.last_producers = None
        self.poll_hold = False
        self.memcpys = []
        self.cancelled = []

    def memcpy_async(self, dst, src, nbytes, stream, kind):
        self.memcpys.append((int(dst), int(src), int(nbytes), stream, int(kind)))
        if int(nbytes) <= 0 or int(kind) not in (1, 2) or not stream:
            return 2
        return 0

    def event_create(self):
        with self.lock:
            ev = self.next_ev
            self.next_ev += 1
            self.events[ev] = 0
            return ev

    def event_record(self, ev, stream=None):
        del stream
        with self.lock:
            self.events[ev] = 1

    def event_query(self, ev):
        with self.lock:
            return int(self.events.get(ev, 0))

    def request_cancel(self, job):
        if isinstance(job, dict):
            self.cancelled.append(int(job.get("sequence", 0)))
        else:
            self.cancelled.append(job)

    def submit_h2d(self, job, spans, epoch, generation):
        del epoch, generation
        self.h2d_spans.append([dict(span) for span in spans])
        self.call_threads.append(threading.get_ident())
        status = 3 if self.fail_h2d else 0
        rc = int(self.h2d_rcs.pop(0)) if self.h2d_rcs else 0
        with self.lock:
            self.submits.append(("h2d", job["sequence"]))
            if rc == 0 and not self.hold:
                self.done.append({"sequence": job["sequence"], "status": status, "direction": 1})
        hook = self.h2d_hook
        if hook is not None and rc == 0:
            self.h2d_hook = None
            hook()
        return rc

    def submit_d2h(self, job, spans, epoch, generation, producers):
        del epoch, generation
        self.call_threads.append(threading.get_ident())
        producers = list(producers or [])
        self.last_producers = producers
        with self.lock:
            self.spans.append([dict(span) for span in spans])
            self.submits.append(("d2h", job["sequence"]))
            item = {"sequence": job["sequence"], "status": 0, "direction": 2, "producers": producers}
            if self._ready(producers):
                self.done.append(item)
            else:
                self.waiting.append(item)
        return 0

    def _ready(self, producers):
        for producer in producers:
            if producer and self.events.get(producer, 0) != 1:
                return False
        return True

    def poll(self, limit=16):
        with self.lock:
            if self.poll_hold:
                return []
            still = []
            for item in self.waiting:
                if self._ready(item["producers"]):
                    self.done.append(
                        {"sequence": item["sequence"], "status": item["status"], "direction": item["direction"]}
                    )
                else:
                    still.append(item)
            self.waiting = still
            out = self.done[: int(limit)]
            del self.done[: int(limit)]
            return out

    def device_sync(self):
        self.syncs += 1
        raise RuntimeError("device sync")


class _FakeStream:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def wait_stream(self, other):
        self.owner.order.append(("wait", other))

    def wait_event(self, event):
        self.owner.order.append(("wait_event", event))


class _FakeStreamCtx:
    def __init__(self, owner, stream):
        self.owner = owner
        self.stream = stream

    def __enter__(self):
        self.owner.active = self.stream
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        self.owner.active = None
        return False


class _FakeTensor:
    def __init__(self, owner, ptr, nbytes, device):
        self.owner = owner
        self.ptr = ptr
        self.nbytes = nbytes
        self.device = device

    def copy_(self, src, non_blocking=False):
        self.owner.order.append(("copy", self.owner.active, bool(non_blocking), src.ptr, self.ptr))


class _FakeEvent:
    def __init__(self, owner):
        self.owner = owner
        self.recorded = None

    def record(self, stream):
        self.recorded = stream
        self.owner.order.append(("record", stream))

    def query(self):
        return not self.owner.block_query


class _FakeCuda:
    def __init__(self, owner):
        self.owner = owner

    def is_available(self):
        return True

    def Stream(self):
        stream = _FakeStream(self.owner, "store")
        stream.cuda_stream = 0x5100
        self.owner.store = stream
        return stream

    def current_stream(self):
        return self.owner.compute

    def stream(self, stream):
        return _FakeStreamCtx(self.owner, stream)

    def Event(self, enable_timing=False):
        del enable_timing
        event = _FakeEvent(self.owner)
        self.owner.events.append(event)
        return event


class FakeTorch:
    """Stand-in for torch.cuda.Stream / Event / copy_ used by D2H tests."""

    def __init__(self):
        self.order = []
        self.events = []
        self.active = None
        self.block_query = False
        self.compute = _FakeStream(self, "compute")
        self.store = None
        self.cuda = _FakeCuda(self)
        self.uint8 = "u8"

    def byte_view(self, ptr, nbytes, device):
        return _FakeTensor(self, int(ptr), int(nbytes), device)

    def device(self, name):
        return name


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.geo = KvGeometry(100, 16, 4, 2, 2)
        self.sdk = FakeSdk()
        self.native = FakeNative()
        slot = self.geo.kvb_bytes(16)
        self.preload_pool = locked_pool(slot, 4, kind="preload")
        self.store_pool = locked_pool(slot, 4, kind="store")
        self.pool = self.store_pool
        self.rt = WorkerRuntime(
            sdk=self.sdk,
            native=self.native,
            policy="finish_drop",
            preload_pool=self.preload_pool,
            store_pool=self.store_pool,
        )
        self.rt.set_geometry(self.geo)
        self.fake_torch = FakeTorch()
        self.rt._torch = self.fake_torch
        self.rt.start()

    def _arm(self, rt):
        rt._torch = self.fake_torch
        return rt

    def tearDown(self):
        self.rt.shutdown()
        self.preload_pool.close()
        self.store_pool.close()

    def test_miss_stays_queued(self):
        self.sdk.scripted[1] = 0
        self.rt.on_new_request("m", [1, 2, 3])
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("m") == (0, False)))
        entry = self.rt.queue.get("m")
        self.assertEqual(entry.status, STATUS_FALLBACK_MISS)

    def test_stale_lookup_cannot_overwrite(self):
        gate = threading.Event()
        self.sdk.gates[1] = gate

        def lookup_hit(req_id, tokens, block_tokens, generation):
            del req_id, block_tokens
            self.sdk.entered.append(int(generation))
            gate.wait(2)
            return 32 if int(tokens[0]) == 1 else 0

        self.sdk.lookup_hit = lookup_hit
        self.rt.on_new_request("s", [1] * 32)
        self.assertTrue(wait_until(lambda: self.sdk.entered == [1]))
        self.rt.on_new_request("s", [2] * 32)
        gate.set()
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("s") == (0, False)))
        self.assertEqual(self.rt.queue.get("s").status, STATUS_FALLBACK_MISS)
        self.assertGreaterEqual(self.rt.stale_drops, 1)

    def test_positive_tokens_only_after_preload(self):
        self.sdk.read_gate = threading.Event()
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("h", [1] * 16)
        self.assertTrue(wait_until(lambda: len(self.sdk.reads) >= 1))
        self.assertEqual(self.rt.matched_tokens("h"), (None, False))
        self.sdk.read_gate.set()
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("h") == (16, True)))
        self.rt.observe_layer("0", 1000, None)
        self.rt.observe_layer("1", 2000, None)
        self.rt.update_state_after_alloc("h", [0], 16)
        self.assertTrue(wait_until(lambda: "h" in self.rt.transfer_results().finished_recving))

    def test_h2d_completion_marks_the_whole_request(self):
        self.rt.shutdown()
        generation = self.rt.queue.enqueue_lookup("p", {"tokens": [1] * 16}, 16)
        self.assertTrue(self.rt.queue.complete_lookup("p", generation, 32, 32))
        self.assertTrue(self.rt.queue.mark_reading("p", generation))
        self.assertTrue(self.rt.queue.mark_preloaded("p", generation, 32))
        with self.rt._mu:
            self.rt._reqs["p"] = _Req(
                req_id="p",
                generation=generation,
                tokens=[1] * 16,
                phase=PHASE_H2D,
                block_ids=[7, 8],
            )
        self.rt._jobs[11] = (CopyKind.H2D, "p", generation, None, {"sequence": 11})
        with self.native.lock:
            self.native.done.append({"sequence": 11, "status": 0, "direction": 1})
        self.rt._poll_copies()
        self.assertIsNone(self.rt.queue.get("p"))
        self.assertNotIn(11, self.rt._jobs)
        with self.rt._mu:
            self.assertEqual(self.rt._reqs["p"].phase, PHASE_HBM)
        miss_gen = self.rt.queue.enqueue_lookup("q", {"tokens": [1] * 16}, 16)
        self.assertTrue(self.rt.queue.complete_lookup("q", miss_gen, 32, 32))
        self.assertTrue(self.rt.queue.mark_reading("q", miss_gen))
        self.assertTrue(self.rt.queue.mark_preloaded("q", miss_gen, 32))
        with self.rt._mu:
            self.rt._reqs["q"] = _Req(
                req_id="q",
                generation=miss_gen,
                tokens=[1] * 16,
                phase=PHASE_H2D,
                block_ids=[7, 8],
            )
        self.rt._jobs[12] = (CopyKind.H2D, "q", miss_gen, None, {"sequence": 12})
        with self.native.lock:
            self.native.done.append({"sequence": 12, "status": 1, "direction": 1})
        self.rt._poll_copies()
        self.assertEqual(self.rt.queue.get("q").status, STATUS_FALLBACK_MISS)
        with self.rt._mu:
            self.assertEqual(self.rt._reqs["q"].phase, PHASE_MISS)

    def test_h2d_failure_is_failed_recving(self):
        board = StatusBoard()
        self.rt.board = board
        self.rt._slots["f"] = 0
        self.native.fail_h2d = True
        self.sdk.scripted[1] = 16
        try:
            self.rt.on_new_request("f", [1] * 16)
            self.assertTrue(wait_until(lambda: self.rt.matched_tokens("f") == (16, True)))
            self.rt.observe_layer("0", 1000, None)
            self.rt.observe_layer("1", 2000, None)
            self.rt.update_state_after_alloc("f", [3], 16)
            self.assertTrue(wait_until(lambda: "f" in self.rt.transfer_results().failed_recving))
            self.assertNotIn("f", self.rt.transfer_results().finished_recving)
            self.assertEqual(self.rt.queue.get("f").status, STATUS_FALLBACK_MISS)
            with self.rt._mu:
                self.assertEqual(self.rt._reqs["f"].phase, PHASE_MISS)
            generation = int(self.rt._reqs["f"].generation)
            self.assertEqual(board.get_at(0).phase, PHASE_MISS)
            self.assertEqual(board.get_at(0).generation, generation)
        finally:
            board.close()

    def test_save_hooks_do_not_store_or_sync_before_d2h(self):
        gen = self.rt.on_new_request("w", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("w") == (0, False)))
        ev = self.native.event_create()
        self.rt.note_block("w", gen, 7, 16, {0: 100}, {0: ev})
        self.rt.save_layer_event("layer0", 1000, ev)
        self.rt.wait_for_save(0)
        time.sleep(0.15)
        sending, _recving = self.rt.get_finished()
        self.assertNotIn("w", sending)
        self.assertEqual(self.native.syncs, 0)
        self.assertEqual(self.fake_torch.order[0][0], "wait")
        self.assertFalse(any(item[0] == "copy" for item in self.fake_torch.order))
        self.assertGreaterEqual(len(self.native.memcpys), 1)
        _dst, _src, _nbytes, stream, kind = self.native.memcpys[0]
        self.assertEqual(stream, self.fake_torch.store.cuda_stream)
        self.assertEqual(kind, 2)
        self.assertTrue(self.rt.request_finished("w", [7], 16))
        self.assertTrue(wait_until(lambda: "w" in self.rt.get_finished()[0]))
        self.assertIn("w", self.rt.transfer_results(["w"]).finished_sending)
        self.assertEqual(self.native.syncs, 0)

    def test_end_mode_records_one_event_and_does_not_wait_stream(self):
        self.rt.shutdown()
        self.rt = self._arm(
            WorkerRuntime(
                sdk=self.sdk,
                native=self.native,
                policy="finish_drop",
                save_mode="end",
                preload_pool=self.preload_pool,
                store_pool=self.store_pool,
            )
        )
        self.rt.set_geometry(self.geo)
        self.rt.start()
        gen = self.rt.on_new_request("end", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("end") == (0, False)))
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        self.rt.note_block("end", gen, 7, 16, {0: 100, 1: 200}, {0: None, 1: None})
        self.assertEqual(self.native.next_ev, 1)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.native.memcpys) >= 1))
        records = [item for item in self.fake_torch.order if item[0] == "record"]
        self.assertGreaterEqual(len(records), 1)
        self.assertIs(records[0][1], self.fake_torch.compute)
        self.assertFalse(any(item[0] == "wait" for item in self.fake_torch.order))
        waited = [item for item in self.fake_torch.order if item[0] == "wait_event"]
        self.assertEqual(len(waited), 1)
        self.assertIs(waited[0][1], self.fake_torch.events[0])
        self.assertEqual(self.native.next_ev, 1)
        self.assertEqual(self.native.syncs, 0)
        self.assertGreaterEqual(len(self.rt.d2h_copies), 1)

    def test_end_mode_copies_a_full_block_then_the_finished_tail(self):
        self.rt.shutdown()
        self.rt = self._arm(
            WorkerRuntime(
                sdk=self.sdk,
                native=self.native,
                policy="finish_drop",
                save_mode="end",
                preload_pool=self.preload_pool,
                store_pool=self.store_pool,
            )
        )
        self.rt.set_geometry(self.geo)
        self.rt.start()
        gen = self.rt.on_new_request("grow", [1] * 20)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("grow") == (0, False)))
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        self.rt.ingest_blocks("grow", gen, [7], 4)
        self.rt.wait_for_save(0)
        with self.rt._mu:
            block = self.rt._reqs["grow"].blocks[7]
            self.assertEqual(block.valid_tokens, 4)
            self.assertFalse(block.inflight)
            self.assertEqual(len(self.rt._reqs["grow"].open_blocks), 1)
        self.assertEqual(self.rt.d2h_copies, [])
        self.rt.ingest_blocks("grow", gen, [7, 8], 20)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) >= 1))
        self.assertTrue(self.rt.request_finished("grow", [7, 8], 20))
        self.assertTrue(wait_until(lambda: any(item[0] == "grow" and item[2] == 4 for item in self.sdk.saves)))
        self.assertFalse(any(item[0] == "grow" for item in self.rt.drops))
        self.assertTrue(wait_until(lambda: "grow" in self.rt.get_finished()[0]))
        self.assertEqual(self.native.syncs, 0)

    def test_unfinished_missing_event_does_not_submit(self):
        gen = self.rt.on_new_request("miss", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("miss") == (0, False)))
        self.rt.note_block("miss", gen, 3, 16, {0: 100}, {0: None})
        before = len(self.rt.d2h_copies)
        self.rt.wait_for_save(0)
        time.sleep(0.15)
        self.assertEqual(len(self.rt.d2h_copies), before)
        with self.rt._mu:
            self.assertFalse(self.rt._reqs["miss"].blocks[3].inflight)

    def test_finished_null_producer_enters_sending(self):
        gen = self.rt.on_new_request("fin", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("fin") == (0, False)))
        self.rt.note_block("fin", gen, 9, 16, {0: 100}, {0: None})
        before = len(self.native.memcpys)
        self.assertTrue(self.rt.request_finished("fin", [9], 16))
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.native.memcpys) > before))
        self.assertTrue(wait_until(lambda: "fin" in self.rt.get_finished()[0]))
        self.assertFalse(any(item[0] == "fin" for item in self.rt.drops))
        self.assertEqual(self.native.syncs, 0)
        self.assertIn("fin", self.rt.transfer_results(["fin"]).finished_sending)

    def _one_slot(self, policy):
        self.rt.shutdown()
        self.pool.close()
        self.preload_pool.close()
        slot = self.geo.kvb_bytes(16)
        self.pool = locked_pool(slot, 1, kind="store")
        self.preload_pool = locked_pool(slot, 2, kind="preload")
        self.store_pool = self.pool
        self.rt = self._arm(
            WorkerRuntime(
                sdk=self.sdk,
                native=self.native,
                policy=policy,
                preload_pool=self.preload_pool,
                store_pool=self.pool,
            )
        )
        self.rt.set_geometry(self.geo)
        self.rt.start()
        return self.pool.try_allocate(slot, "store", "other", 1)

    def test_finished_sibling_drops_without_another_save(self):
        held = self._one_slot("finish_drop")
        self.assertIsNotNone(held)
        gen = self.rt.on_new_request("pair", [1] * 32)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("pair") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("pair", gen, 1, 16, {0: 5}, {0: ev}, token_index=0)
        self.rt.note_block("pair", gen, 2, 16, {0: 6}, {0: ev}, token_index=1)
        with self.rt._mu:
            done = self.rt._reqs["pair"].blocks[1]
            done.saved = True
            done.d2h_done = True
        self.assertTrue(self.rt.request_finished("pair", [1, 2], 32))
        self.assertTrue(wait_until(lambda: "pair" in self.rt.get_finished()[0]))
        self.assertTrue(any(item[0] == "pair" and item[2] == 2 for item in self.rt.drops))
        self.pool.abort_unstored(held)

    def test_finished_orphan_inflight_is_dropped(self):
        gen = self.rt.on_new_request("orphan", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("orphan") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("orphan", gen, 4, 16, {0: 5}, {0: ev})
        with self.rt._mu:
            self.rt._reqs["orphan"].blocks[4].inflight = True
        self.assertTrue(self.rt.request_finished("orphan", [4], 16))
        self.assertIn("orphan", self.rt.get_finished()[0])
        self.assertTrue(any(item[0] == "orphan" and item[2] == 4 for item in self.rt.drops))
        self.assertEqual(self.rt.d2h_copies, [])

    def test_finished_copy_holds_until_the_event(self):
        gen = self.rt.on_new_request("room", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("room") == (0, False)))
        self.fake_torch.block_query = True
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("room", gen, 8, 16, {0: 5}, {0: ev})
        before = len(self.rt.d2h_copies)
        self.assertTrue(self.rt.request_finished("room", [8], 16))
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) > before))
        self.assertNotIn("room", self.rt.get_finished()[0])
        self.assertFalse(any(item[0] == "room" for item in self.rt.drops))
        self.fake_torch.block_query = False
        self.assertTrue(wait_until(lambda: "room" in self.rt.get_finished()[0]))
        self.assertEqual(self.native.syncs, 0)

    def test_finished_queued_block_is_copied(self):
        gen = self.rt.on_new_request("queued", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("queued") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("queued", gen, 6, 16, {0: 5}, {0: ev})
        self.rt.wait_for_save(0)
        self.assertTrue(self.rt.request_finished("queued", [6], 16))
        self.assertFalse(any(item[0] == "queued" for item in self.rt.drops))
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) >= 1))
        self.assertTrue(wait_until(lambda: "queued" in self.rt.get_finished()[0]))
        self.assertEqual(self.native.syncs, 0)

    def test_inflight_copy_does_not_hold_another_request(self):
        gen = self.rt.on_new_request("fly", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("fly") == (0, False)))
        self.fake_torch.block_query = True
        ev = self.fake_torch.cuda.Event()
        self.rt.note_block("fly", gen, 8, 16, {0: 5}, {0: ev})
        slots = self.pool.try_allocate(self.geo.kvb_bytes(16), "store", "fly", gen)
        self.assertIsNotNone(slots)
        with self.rt._mu:
            block = self.rt._reqs["fly"].blocks[8]
            block.inflight = True
            self.rt._d2h_inflight.append(
                {
                    "event": ev,
                    "kept": None,
                    "req_id": "fly",
                    "generation": gen,
                    "slots": slots,
                    "block": block,
                }
            )
        self.assertTrue(self.rt.request_finished("fly", [8], 16))
        self.assertNotIn("fly", self.rt.get_finished()[0])
        self.assertFalse(any(item[0] == "fly" for item in self.rt.drops))

        self.rt.shutdown()
        gen2 = self.rt.on_new_request("side", [2] * 16)
        ev2 = self.native.event_create()
        self.native.event_record(ev2)
        self.rt.note_block("side", gen2, 3, 16, {0: 9}, {0: ev2})
        with self.rt._mu:
            side = self.rt._reqs["side"].blocks[3]
            side.inflight = True
            self.rt._snaps.append([("side", gen2, side)])
        before = len(self.rt.d2h_copies)
        self.assertTrue(self.rt.request_finished("side", [3], 16))
        self.assertNotIn("side", self.rt.get_finished()[0])
        self.assertNotIn("fly", self.rt.get_finished()[0])
        self.assertFalse(any(item[0] == "side" for item in self.rt.drops))
        self.assertEqual(len(self.rt.d2h_copies), before)
        with self.rt._mu:
            still = [
                int(block.block_id)
                for snap in self.rt._snaps
                for req_id, _generation, block in snap
                if req_id == "side"
            ]
        self.assertEqual(still, [3])

        self.fake_torch.block_query = False

        def fly_saved():
            self.rt._poll_copies()
            with self.rt._mu:
                item = self.rt._store.pop(0) if self.rt._store else None
            if item is not None:
                _req_id, _generation, _slots, block = item[:4]
                self.sdk.enqueue_save(
                    _req_id,
                    _generation,
                    block.valid_tokens,
                    self.geo.block_size,
                    _slots,
                    int(block.token_index),
                    lambda status: None,
                )
                with self.rt._mu:
                    req = self.rt._reqs.get(str(_req_id))
                    if req is not None:
                        self.rt._release_saved_locked(req)
            return "fly" in self.rt.get_finished()[0]

        self.assertTrue(wait_until(fly_saved))
        self.assertTrue(any(item[0] == "fly" for item in self.sdk.saves))
        self.assertFalse(any(item[0] == "fly" for item in self.rt.drops))
        self.rt.clear_published()
        self.assertNotIn("fly", self.rt.get_finished()[0])
        self.assertEqual(self.native.syncs, 0)

    def test_poll_query_does_not_drop_the_inflight_block(self):
        gen = self.rt.on_new_request("gap", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("gap") == (0, False)))
        ev = self.fake_torch.cuda.Event()
        self.rt.note_block("gap", gen, 8, 16, {0: 5}, {0: ev})
        slots = self.pool.try_allocate(self.geo.kvb_bytes(16), "store", "gap", gen)
        self.assertIsNotNone(slots)
        entered = {}

        def query():
            if not entered.get("ran"):
                entered["ran"] = True
                self.rt.request_finished("gap", [8], 16)
                self.assertFalse(any(item[0] == "gap" for item in self.rt.drops))
                with self.rt._mu:
                    self.assertIn(("gap", int(gen), 8), self.rt._d2h_handed)
                    self.assertNotIn("gap", self.rt._sending)
            return False

        ev.query = query
        with self.rt._mu:
            block = self.rt._reqs["gap"].blocks[8]
            block.inflight = True
            self.rt._d2h_inflight.append(
                {
                    "event": ev,
                    "kept": None,
                    "req_id": "gap",
                    "generation": gen,
                    "slots": slots,
                    "block": block,
                }
            )
        self.assertNotIn("gap", self.rt.get_finished()[0])
        self.assertTrue(entered.get("ran"))
        with self.rt._mu:
            self.assertEqual(len(self.rt._d2h_inflight), 1)
            self.assertNotIn(("gap", int(gen), 8), self.rt._d2h_handed)

    def test_live_full_pool_is_not_dropped(self):
        held = self._one_slot("finish_drop")
        self.assertIsNotNone(held)
        gen = self.rt.on_new_request("livefull", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("livefull") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("livefull", gen, 3, 16, {0: 5}, {0: ev})
        self.rt.wait_for_save(0)

        def inflight():
            with self.rt._mu:
                return self.rt._reqs["livefull"].blocks[3].inflight

        self.assertTrue(wait_until(lambda: not inflight()))
        self.assertFalse(any(item[0] == "livefull" for item in self.rt.drops))
        self.assertNotIn("livefull", self.rt.get_finished()[0])
        self.pool.abort_unstored(held)

    def test_finish_drop_live_retries_on_next_save(self):
        held = self._one_slot("finish_drop")
        gen = self.rt.on_new_request("hold", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("hold") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("hold", gen, 1, 16, {0: 5}, {0: ev})
        before = len(self.rt.d2h_copies)
        self.rt.wait_for_save(0)

        def inflight():
            with self.rt._mu:
                return self.rt._reqs["hold"].blocks[1].inflight

        self.assertTrue(wait_until(lambda: not inflight()))
        self.assertEqual(len(self.rt.d2h_copies), before)
        self.assertFalse(any(item[0] == "hold" for item in self.rt.drops))
        self.assertNotIn("hold", self.rt.get_finished()[0])
        self.pool.abort_unstored(held)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) == before + 1))

    def test_finish_drop_after_finished_drops_without_save(self):
        self._one_slot("finish_drop")
        gen = self.rt.on_new_request("hold", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("hold") == (0, False)))
        self.rt.note_block("hold", gen, 1, 16, {0: 5}, {0: None})
        self.assertTrue(self.rt.request_finished("hold", [1], 16))
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: "hold" in self.rt.get_finished()[0]))
        self.assertTrue(any(item[0] == "hold" for item in self.rt.drops))
        self.assertEqual(self.rt.d2h_copies, [])
        self.assertEqual(self.sdk.saves, [])

    def test_drop_before_finish_then_sending(self):
        self._one_slot("drop")
        gen = self.rt.on_new_request("early", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("early") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("early", gen, 4, 16, {0: 5}, {0: ev})
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: any(item[0] == "early" for item in self.rt.drops)))
        self.assertNotIn("early", self.rt.get_finished()[0])
        self.assertTrue(self.rt.request_finished("early", [4], 16))
        self.assertIn("early", self.rt.get_finished()[0])
        self.assertEqual(self.sdk.saves, [])

    def test_stale_finish_does_not_enter_sending(self):
        gen = self.rt.on_new_request("gone", [1] * 16)
        self.rt.cancel("gone")
        self.assertFalse(self.rt.request_finished("gone", [0, 1, 2], 48, generation=gen))
        self.assertNotIn("gone", self.rt.get_finished()[0])
        with self.rt._mu:
            self.assertFalse(self.rt._reqs["gone"].finished)
            self.assertTrue(self.rt._reqs["gone"].closed)
        gen = self.rt.on_new_request("old", [1] * 16)
        self.assertFalse(self.rt.request_finished("old", [0], 16, generation=gen + 1))
        self.assertNotIn("old", self.rt.get_finished()[0])
        with self.rt._mu:
            self.assertFalse(self.rt._reqs["old"].finished)
            self.assertEqual(self.rt._reqs["old"].generation, gen)

    def test_shorter_ingest_reclips_hold_and_can_send(self):
        gen = self.rt.on_new_request("shrink", [1] * 48)
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        with self.rt._mu:
            for block_id in (1, 2):
                self.rt._d2h_handed.add(("shrink", gen, block_id))
        self.assertTrue(self.rt.request_finished("shrink", [0, 1, 2], 48, generation=gen))
        self.assertNotIn("shrink", self.rt.get_finished()[0])
        self.rt.ingest_blocks("shrink", gen, [0, 1, 2], 16)
        self.assertNotIn("shrink", self.rt.get_finished()[0])
        with self.rt._mu:
            req = self.rt._reqs["shrink"]
            owned = [req.blocks[1], req.blocks[2]]
        slots = self.pool.try_allocate(self.geo.kvb_bytes(16) * 2, "store", "shrink", gen)
        self.rt._apply_d2h_result(
            {
                "req_id": "shrink",
                "generation": gen,
                "slots": slots,
                "status": 0,
                "block": owned[0],
                "blocks": owned,
                "group": {
                    "slots": slots,
                    "left": 2,
                    "failed": False,
                    "count": 2,
                    "lock": threading.Lock(),
                },
            }
        )
        self.assertTrue(wait_until(lambda: "shrink" in self.rt.get_finished()[0]))
        self.assertFalse(any(item[0] == "shrink" and item[3] in (1, 2) for item in self.sdk.saves))

    def test_shorter_ingest_drops_queued_tail(self):
        gen = self.rt.on_new_request("tailq", [1] * 48)
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        self.rt.ingest_blocks("tailq", gen, [0, 1, 2], 48)
        with self.rt._mu:
            req = self.rt._reqs["tailq"]
            req.finished = True
            req.blocks[0].d2h_done = True
            req.blocks[0].saved = True
            self.rt._snaps.append([("tailq", gen, req.blocks[1]), ("tailq", gen, req.blocks[2])])
        self.rt.ingest_blocks("tailq", gen, [0, 1, 2], 16)
        with self.rt._mu:
            req = self.rt._reqs["tailq"]
            queued = [
                int(block.block_id)
                for snap in self.rt._snaps
                for req_id, _gen, block in snap
                if str(req_id) == "tailq"
            ]
            self.assertEqual(queued, [])
            self.assertTrue(req.blocks[1].d2h_done)
            self.assertTrue(req.blocks[2].d2h_done)
        self.assertIn("tailq", self.rt.get_finished()[0])
        self.assertFalse(any(item[0] == "tailq" for item in self.sdk.saves))

    def test_second_request_finished_does_not_release_again(self):
        class _Board:
            def __init__(self):
                self.releases = []

            def release(self, slot, generation):
                self.releases.append((int(slot), int(generation)))

        board = _Board()
        self.rt.board = board
        gen = self.rt.on_new_request("dup", [1] * 16)
        with self.rt._mu:
            self.rt._slots["dup"] = 3
        self.assertTrue(self.rt.request_finished("dup", [0], 16, slot=3, generation=gen))
        self.assertEqual(board.releases, [(3, gen)])
        self.assertTrue(self.rt.request_finished("dup", [0], 8, slot=3, generation=gen))
        self.assertEqual(board.releases, [(3, gen)])

    def test_hold_skips_blocks_past_num_tokens(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        self.rt.on_new_request("clip", [1] * 32)
        self.assertTrue(self.rt.request_finished("clip", [0, 1, 2], 16))
        self.assertTrue(wait_until(lambda: "clip" in self.rt.get_finished()[0]))
        self.assertFalse(any(item[0] == "clip" and item[3] in (1, 2) for item in self.sdk.saves))
        with self.rt._mu:
            self.assertNotIn(1, self.rt._reqs["clip"].blocks)
            self.assertNotIn(2, self.rt._reqs["clip"].blocks)

    def test_new_request_copies_layer_bases(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        gen = self.rt.on_new_request("nb", [1] * 16)
        self.rt.ingest_blocks("nb", gen, [4], 16)
        stride = self.geo.layer_stride()
        with self.rt._mu:
            block = self.rt._reqs["nb"].blocks[4]
            self.assertEqual(block.src["0"], 1000 + 4 * stride)
            self.assertEqual(block.src["1"], 2000 + 4 * stride)

    def test_incremental_materialize_keeps_src(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        gen = self.rt.on_new_request("inc", [1] * 40)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("inc") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.observe_layer("0", 1000, ev)
        self.rt.observe_layer("1", 2000, ev)
        self.rt.ingest_blocks("inc", gen, [10, 11, 12], 36)
        stride = self.geo.layer_stride()
        with self.rt._mu:
            req = self.rt._reqs["inc"]
            self.assertEqual(req.blocks[10].valid_tokens, 16)
            self.assertEqual(req.blocks[12].valid_tokens, 4)
            self.assertEqual(len(req.open_blocks), 3)
            src_full = id(req.blocks[10].src)
            src_mid = id(req.blocks[11].src)
            src_tail = id(req.blocks[12].src)
        before = len(self.rt.d2h_copies)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) >= before + 1))
        grouped = self.rt.d2h_copies[-1]
        self.assertEqual(len(grouped), 2)
        self.assertEqual({span["bytes"] for span in grouped}, {2 * self.geo.layer_payload(16)})
        self.rt.ingest_blocks("inc", gen, [10, 11, 12], 40)
        self.rt.wait_for_save(0)
        time.sleep(0.2)
        with self.rt._mu:
            req = self.rt._reqs["inc"]
            self.assertEqual(id(req.blocks[10].src), src_full)
            self.assertEqual(id(req.blocks[11].src), src_mid)
            self.assertEqual(id(req.blocks[12].src), src_tail)
            self.assertEqual(req.blocks[12].valid_tokens, 8)
            self.assertEqual(req.blocks[10].src["0"], 1000 + 10 * stride)
            open_ids = [int(block.block_id) for block in req.open_blocks]
            self.assertNotIn(10, open_ids)
            self.assertNotIn(11, open_ids)
            self.assertEqual(open_ids.count(12), 1)
        self.assertEqual(len(self.rt.d2h_copies), before + 1)
        self.rt.set_layer_bases({"0": 5000, "1": 6000})
        self.rt.wait_for_save(0)
        with self.rt._mu:
            req = self.rt._reqs["inc"]
            self.assertEqual(id(req.blocks[12].src), src_tail)
            self.assertEqual(req.blocks[12].src["0"], 5000 + 12 * stride)
            self.assertEqual(req.blocks[10].src["0"], 1000 + 10 * stride)
        self.assertEqual(len(self.rt.d2h_copies), before + 1)

    def test_open_tail_takes_the_latest_event(self):
        self.rt.set_layer_bases({"0": 1000})
        gen = self.rt.on_new_request("step", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("step") == (0, False)))
        ev1 = self.native.event_create()
        self.native.event_record(ev1)
        self.rt.observe_layer("0", 1000, ev1)
        self.rt.ingest_blocks("step", gen, [5], 4)
        self.rt.wait_for_save(0)
        with self.rt._mu:
            block = self.rt._reqs["step"].blocks[5]
            self.assertFalse(block.inflight)
            self.assertIs(block.events["0"], ev1)
            events_id = id(block.events)
        ev2 = self.native.event_create()
        self.native.event_record(ev2)
        self.rt.observe_layer("0", 1000, ev2)
        self.rt.ingest_blocks("step", gen, [5], 16)
        self.rt.wait_for_save(0)
        with self.rt._mu:
            block = self.rt._reqs["step"].blocks[5]
            self.assertIs(block.events["0"], ev2)
            self.assertEqual(id(block.events), events_id)
            self.assertTrue(block.inflight or block.d2h_done)
            open_ids = [int(item.block_id) for item in self.rt._reqs["step"].open_blocks]
            self.assertNotIn(5, open_ids)

    def test_global_table_beats_a_stale_request_copy(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        epoch = self.rt._base_epoch
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        self.assertEqual(self.rt._base_epoch, epoch)
        gen = self.rt.on_new_request("g", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("g") == (0, False)))
        ev1 = self.native.event_create()
        self.native.event_record(ev1)
        self.rt.observe_layer("0", 1000, ev1)
        self.rt.observe_layer("1", 2000, ev1)
        self.rt.ingest_blocks("g", gen, [5], 4)
        self.rt.wait_for_save(0)
        stride = self.geo.layer_stride()
        with self.rt._mu:
            req = self.rt._reqs["g"]
            self.assertEqual(req.layer_base, {})
            req.layer_base = {"0": 9, "1": 9}
            req.layer_events = {"0": "stale", "1": "stale"}
            req.base_epoch = 99
            self.assertEqual(req.blocks[5].src["0"], 1000 + 5 * stride)
        ev2 = self.native.event_create()
        self.native.event_record(ev2)
        self.rt.observe_layer("0", 4000, ev2)
        self.rt.observe_layer("1", 5000, ev2)
        self.rt.wait_for_save(0)
        with self.rt._mu:
            req = self.rt._reqs["g"]
            block = req.blocks[5]
            self.assertEqual(block.src["0"], 4000 + 5 * stride)
            self.assertEqual(block.src["1"], 5000 + 5 * stride)
            self.assertIs(block.events["0"], ev2)
            self.assertIs(block.events["1"], ev2)
            self.assertEqual(req.layer_base["0"], 9)
            self.assertEqual(req.layer_events["0"], "stale")

    def test_submitted_block_keeps_its_address_and_event(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        gen = self.rt.on_new_request("kept", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("kept") == (0, False)))
        ev1 = self.native.event_create()
        self.native.event_record(ev1)
        self.rt.observe_layer("0", 1000, ev1)
        self.rt.observe_layer("1", 2000, ev1)
        self.rt.ingest_blocks("kept", gen, [5], 16)
        self.native.poll_hold = True
        self.rt.wait_for_save(0)
        stride = self.geo.layer_stride()
        with self.rt._mu:
            block = self.rt._reqs["kept"].blocks[5]
            self.assertTrue(block.inflight or block.d2h_done or block.saved)
            self.assertIs(block.events["0"], ev1)
            self.assertEqual(block.src["0"], 1000 + 5 * stride)
        ev2 = self.native.event_create()
        self.native.event_record(ev2)
        self.rt.observe_layer("0", 7000, ev2)
        self.rt.observe_layer("1", 8000, ev2)
        self.rt.wait_for_save(0)
        with self.rt._mu:
            block = self.rt._reqs["kept"].blocks[5]
            self.assertIs(block.events["0"], ev1)
            self.assertEqual(block.src["0"], 1000 + 5 * stride)
            self.assertNotIn(5, [int(item.block_id) for item in self.rt._reqs["kept"].open_blocks])
        self.native.poll_hold = False

    def test_active_set_leaves_on_seal_finish_and_cancel(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        with self.rt._mu:
            live = _Req(req_id="live", generation=1, tokens=[1] * 16, num_tokens=4, block_ids=[5])
            gone = _Req(req_id="gone", generation=1, tokens=[1] * 16, num_tokens=4, block_ids=[6])
            self.rt._reqs["live"] = live
            self.rt._reqs["gone"] = gone
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.observe_layer("0", 1000, ev)
        self.rt.observe_layer("1", 2000, ev)
        self.rt.ingest_blocks("live", 1, [5], 4)
        self.rt.ingest_blocks("gone", 1, [6], 4)
        self.assertIn("live", self.rt._save_active)
        self.assertIn("gone", self.rt._save_active)
        self.rt.cancel("gone")
        self.assertNotIn("gone", self.rt._save_active)
        with self.rt._mu:
            self.assertTrue(self.rt._reqs["gone"].closed)
        self.rt.ingest_blocks("live", 1, [5], 16)
        self.native.poll_hold = True
        self.rt.wait_for_save(0)
        self.assertNotIn("live", self.rt._save_active)
        with self.rt._mu:
            self.assertNotIn(5, [int(block.block_id) for block in self.rt._reqs["live"].open_blocks])
            hold = _Req(
                req_id="hold",
                generation=1,
                tokens=[1] * 16,
                num_tokens=16,
                built_tokens=16,
                finished=True,
                block_ids=[8],
            )
            self.rt._reqs["hold"] = hold
            self.rt._d2h_handed.add(("hold", 1, 8))
            self.rt._sync_save_active_locked(hold)
        self.assertIn("hold", self.rt._save_active)
        with self.rt._mu:
            self.rt._d2h_handed.discard(("hold", 1, 8))
            self.rt._release_saved_locked(self.rt._reqs["hold"])
        self.assertNotIn("hold", self.rt._save_active)
        self.assertIn("hold", self.rt._sending)
        self.native.poll_hold = False

    def test_observe_and_wait_skip_idle_requests(self):
        self.rt.set_layer_bases({"0": 1000, "1": 2000})
        with self.rt._mu:
            live = _Req(req_id="live", generation=1, tokens=[1] * 16, num_tokens=4, block_ids=[5])
            self.rt._reqs["live"] = live
            for index in range(40):
                key = "idle%d" % index
                self.rt._reqs[key] = _Req(
                    req_id=key, generation=1, tokens=[1] * 16, num_tokens=16, built_tokens=16
                )
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.observe_layer("0", 1000, ev)
        self.rt.observe_layer("1", 2000, ev)
        self.rt.ingest_blocks("live", 1, [5], 4)
        with self.rt._mu:
            for index in range(40):
                idle = self.rt._reqs["idle%d" % index]
                self.assertEqual(idle.layer_base, {})
                self.assertEqual(idle.layer_events, {})
                self.assertEqual(idle.base_epoch, 0)
            self.assertIs(self.rt._layer_events["0"], ev)
        seen = []
        orig = self.rt._materialize_locked

        def wrapped(req, block_ids):
            seen.append(req.req_id)
            return orig(req, block_ids)

        self.rt._materialize_locked = wrapped
        self.rt.wait_for_save(0)
        self.assertEqual(seen, ["live"])
        self.assertEqual(self.rt._save_active, {"live"})

    def test_scan_fields_append_to_the_decode_trace_line(self):
        import surdata_vllm.streams as streams_mod

        lines = []

        def _emit(msg):
            lines.append(msg)

        fake = type("Trace", (), {"_emit": staticmethod(_emit)})()
        previous = sys.modules.get("surdata_lmcache.decode_trace")
        sys.modules["surdata_lmcache.decode_trace"] = fake
        old_flag = streams_mod._trace_emit_patched
        old_suffix = streams_mod._trace_suffix
        old_env = os.environ.get("SURDATA_DECODE_TRACE")
        streams_mod._trace_emit_patched = False
        streams_mod._trace_suffix = None
        os.environ["SURDATA_DECODE_TRACE"] = "1"
        try:
            self.rt.set_layer_bases({"0": 1000, "1": 2000})
            with self.rt._mu:
                self.rt._reqs["live"] = _Req(
                    req_id="live", generation=1, tokens=[1] * 16, num_tokens=4, block_ids=[5]
                )
                for index in range(30):
                    key = "idle%d" % index
                    self.rt._reqs[key] = _Req(
                        req_id=key, generation=1, tokens=[1] * 16, num_tokens=16, built_tokens=16
                    )
            ev = self.native.event_create()
            self.native.event_record(ev)
            self.rt.observe_layer("0", 1000, ev)
            self.rt.observe_layer("1", 2000, ev)
            self.rt.ingest_blocks("live", 1, [5], 4)
            self.rt.wait_for_save(0)
            self.assertTrue(streams_mod._trace_suffix)
            self.assertIn("observe_reqs=0", streams_mod._trace_suffix)
            self.assertIn("save_scan_reqs=1", streams_mod._trace_suffix)
            sys.modules["surdata_lmcache.decode_trace"]._emit(
                "SURDATA_DECODE_TRACE step=4 kind=mixed reqs=159"
            )
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("SURDATA_DECODE_TRACE step=4 "))
            self.assertIn("observe_reqs=0", lines[0])
            self.assertIn("save_scan_reqs=1", lines[0])
            self.assertIn("save_open_blocks=", lines[0])
            self.assertIn("event_create_ms=", lines[0])
            self.assertIsNone(streams_mod._trace_suffix)
        finally:
            if old_env is None:
                os.environ.pop("SURDATA_DECODE_TRACE", None)
            else:
                os.environ["SURDATA_DECODE_TRACE"] = old_env
            if previous is None:
                sys.modules.pop("surdata_lmcache.decode_trace", None)
            else:
                sys.modules["surdata_lmcache.decode_trace"] = previous
            streams_mod._trace_emit_patched = old_flag
            streams_mod._trace_suffix = old_suffix

    def test_scan_fields_print_when_the_trace_module_is_absent(self):
        import surdata_vllm.streams as streams_mod

        previous = sys.modules.pop("surdata_lmcache.decode_trace", None)
        old_flag = streams_mod._trace_emit_patched
        old_suffix = streams_mod._trace_suffix
        old_env = os.environ.get("SURDATA_DECODE_TRACE")
        streams_mod._trace_emit_patched = False
        streams_mod._trace_suffix = None
        os.environ["SURDATA_DECODE_TRACE"] = "1"
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                self.rt.wait_for_save(0)
            text = buf.getvalue()
            self.assertIn("SURDATA_DECODE_TRACE ", text)
            self.assertIn("observe_reqs=0", text)
            self.assertIn("save_scan_reqs=0", text)
        finally:
            if old_env is None:
                os.environ.pop("SURDATA_DECODE_TRACE", None)
            else:
                os.environ["SURDATA_DECODE_TRACE"] = old_env
            if previous is not None:
                sys.modules["surdata_lmcache.decode_trace"] = previous
            streams_mod._trace_emit_patched = old_flag
            streams_mod._trace_suffix = old_suffix

    def test_note_block_replaces_open_entry(self):
        gen = self.rt.on_new_request("nb2", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("nb2") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("nb2", gen, 7, 16, {0: 1}, {0: ev})
        self.rt.note_block("nb2", gen, 7, 16, {0: 2}, {0: ev})
        with self.rt._mu:
            req = self.rt._reqs["nb2"]
            self.assertEqual(len(req.open_blocks), 1)
            self.assertIs(req.open_blocks[0], req.blocks[7])
            self.assertEqual(req.blocks[7].src[0], 2)

    def test_finished_short_tail_is_saved(self):
        gen = self.rt.on_new_request("tail", [1] * 4)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("tail") == (0, False)))
        self.rt.note_block("tail", gen, 9, 4, {"0": 100}, {"0": None})
        self.assertTrue(self.rt.request_finished("tail", [9], 4))
        self.assertTrue(wait_until(lambda: any(item[0] == "tail" and item[2] == 4 for item in self.sdk.saves)))
        self.assertFalse(any(item[0] == "tail" for item in self.rt.drops))
        self.assertGreaterEqual(len(self.rt.d2h_copies), 1)
        self.assertTrue(wait_until(lambda: "tail" in self.rt.get_finished()[0]))

    def test_live_short_tail_waits_for_a_full_block(self):
        gen = self.rt.on_new_request("live", [1] * 4)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("live") == (0, False)))
        self.rt.note_block("live", gen, 3, 4, {"0": 100}, {"0": None})
        self.rt.wait_for_save(0)
        time.sleep(0.15)
        self.assertEqual(self.sdk.saves, [])
        with self.rt._mu:
            block = self.rt._reqs["live"].blocks[3]
            self.assertFalse(block.inflight)
            self.assertFalse(block.d2h_done)
            self.assertFalse(block.saved)

    def test_finished_without_layer_base_counts_as_done(self):
        self.rt.on_new_request("bare", [1] * 16)
        self.assertTrue(self.rt.request_finished("bare", [3], 16))
        self.rt.wait_for_save(0)
        with self.rt._mu:
            req = self.rt._reqs["bare"]
            self.assertNotIn(3, req.blocks)
        self.assertIn("bare", self.rt.get_finished()[0])
        self.assertEqual(self.sdk.saves, [])
        self.assertEqual(self.rt.d2h_copies, [])

    def test_get_finished_polls_first(self):
        gen = self.rt.on_new_request("g", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("g") == (0, False)))
        self.fake_torch.block_query = True
        self.rt.note_block("g", gen, 2, 16, {0: 8}, {0: None})
        self.assertTrue(self.rt.request_finished("g", [2], 16))
        self.rt.wait_for_save(0)
        self.assertNotIn("g", self.rt.get_finished()[0])
        self.assertFalse(any(item[0] == "g" for item in self.rt.drops))
        with self.rt._mu:
            queued = any(
                str(req_id) == "g"
                for snap in self.rt._snaps
                for req_id, _gen, _block in snap
            )
            held = self.rt._reqs["g"].blocks[2].inflight or 2 in {
                int(block_id)
                for req_id, _gen, block_id in self.rt._d2h_handed
                if str(req_id) == "g"
            }
        self.assertTrue(queued or held or self.rt._d2h_inflight or self.rt.d2h_copies)
        self.fake_torch.block_query = False
        self.assertTrue(wait_until(lambda: "g" in self.rt.get_finished()[0]))
        self.assertEqual(self.native.syncs, 0)

    def test_request_finished_after_d2h_done_is_sending(self):
        self.rt.on_new_request("done", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("done") == (0, False)))
        self.assertTrue(self.rt.request_finished("done", [7], 16))
        self.assertIn("done", self.rt.get_finished()[0])

    def test_finished_sending_reported_once(self):
        self.rt.on_new_request("once", [1] * 16)
        self.assertTrue(self.rt.request_finished("once", [3], 16))
        self.rt.wait_for_save(0)
        self.assertIn("once", self.rt.transfer_results().finished_sending)
        self.rt.clear_published()
        self.rt.wait_for_save(0)
        self.assertNotIn("once", self.rt.transfer_results().finished_sending)

    def test_clear_keeps_sending_arrived_after_snapshot(self):
        self.rt.on_new_request("A", [1] * 16)
        self.assertTrue(self.rt.request_finished("A", [1], 16))
        self.rt.wait_for_save(0)
        self.assertIn("A", self.rt.transfer_results().finished_sending)
        with self.rt._mu:
            self.rt._sending.add("B")
        self.rt.clear_published()
        nxt = self.rt.transfer_results()
        self.assertNotIn("A", nxt.finished_sending)
        self.assertIn("B", nxt.finished_sending)

    def test_clear_keeps_recving_arrived_after_snapshot(self):
        with self.rt._mu:
            self.rt._recving.add("A")
        self.assertIn("A", self.rt.transfer_results().finished_recving)
        with self.rt._mu:
            self.rt._recving.add("B")
        self.rt.clear_published()
        nxt = self.rt.transfer_results()
        self.assertNotIn("A", nxt.finished_recving)
        self.assertIn("B", nxt.finished_recving)

    def test_clear_keeps_failed_arrived_after_snapshot(self):
        with self.rt._mu:
            self.rt._failed.add("A")
        self.assertIn("A", self.rt.transfer_results().failed_recving)
        with self.rt._mu:
            self.rt._failed.add("B")
        self.rt.clear_published()
        nxt = self.rt.transfer_results()
        self.assertNotIn("A", nxt.failed_recving)
        self.assertIn("B", nxt.failed_recving)

    def test_overlap_reported_only_as_recving(self):
        self.rt.on_new_request("both", [1] * 16)
        self.assertTrue(self.rt.request_finished("both", [3], 16))
        self.rt.wait_for_save(0)
        with self.rt._mu:
            self.assertIn("both", self.rt._sending)
            self.rt._recving.add("both")
        got = self.rt.transfer_results()
        self.assertNotIn("both", got.finished_sending)
        self.assertIn("both", got.finished_recving)
        self.rt.clear_published()
        self.rt.wait_for_save(0)
        self.assertNotIn("both", self.rt.transfer_results().finished_sending)

    def test_zero_external_tokens_still_submits_d2h(self):
        self.rt.on_new_request("m0", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("m0") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.observe_layer("0", 1000, ev)
        self.rt.observe_layer("1", 2000, ev)
        self.assertFalse(self.rt.update_state_after_alloc("m0", [5], 0))
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) >= 1))

    def test_finish_drop_unfinished_retries_after_gc(self):
        held = self._one_slot("finish_drop")
        self.pool.mark_stored(held)
        gen = self.rt.on_new_request("d", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.queue.get("d") is not None))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("d", gen, 1, 16, {0: 5}, {0: ev})
        before = len(self.rt.d2h_copies)
        self.rt.wait_for_save(0)

        def inflight():
            with self.rt._mu:
                return self.rt._reqs["d"].blocks[1].inflight

        self.assertTrue(wait_until(lambda: not inflight()))
        self.assertEqual(len(self.rt.d2h_copies), before)
        self.assertFalse(any(item[0] == "d" for item in self.rt.drops))
        self.assertNotIn("d", self.rt.get_finished()[0])
        self.pool.unpin(held)
        self.pool.request_watermark(self.pool.slot_bytes, self.pool.slot_bytes)
        self.assertTrue(wait_until(lambda: self.pool.occupied() == 0))
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: len(self.rt.d2h_copies) >= 1))

    def test_pool_full_drop_releases_without_d2h(self):
        self.rt.shutdown()
        self.pool.close()
        self.preload_pool.close()
        self.pool = locked_pool(self.geo.kvb_bytes(16), 1, kind="store")
        self.preload_pool = locked_pool(self.geo.kvb_bytes(16), 2, kind="preload")
        self.store_pool = self.pool
        self.rt = self._arm(
            WorkerRuntime(
                sdk=self.sdk,
                native=self.native,
                policy="drop",
                preload_pool=self.preload_pool,
                store_pool=self.pool,
            )
        )
        self.rt.set_geometry(self.geo)
        self.rt.start()
        held = self.pool.try_allocate(self.pool.slot_bytes, "store", "other", 1)
        self.pool.mark_stored(held)
        gen = self.rt.on_new_request("p", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("p") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("p", gen, 4, 16, {0: 5}, {0: ev})
        self.rt.request_finished("p", [4], 16)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: "p" in self.rt.get_finished()[0]))
        self.assertIn("p", self.rt.transfer_results(["p"]).finished_sending)
        self.assertTrue(any(item[0] == "p" for item in self.rt.drops))
        self.assertEqual(self.rt.d2h_copies, [])
        self.assertFalse(any(item[0] == "d2h" for item in self.native.submits))

    def test_gc_evicts_only_stored_unpinned(self):
        pool = locked_pool(100, 3)
        slot_a = pool.try_allocate(100, "store", "a", 1)
        slot_b = pool.try_allocate(100, "store", "b", 1)
        slot_c = pool.try_allocate(100, "store", "c", 1)
        pool.mark_stored(slot_a)
        pool.mark_stored(slot_b)
        pool.unpin(slot_b)
        pool.unpin(slot_c)
        pool.request_watermark(300, 100)
        pool.reclaim_to_watermark()
        self.assertFalse(slot_a[0].freed)
        self.assertTrue(slot_b[0].freed)
        self.assertFalse(slot_c[0].freed)
        self.assertFalse(hasattr(pool, "reclaim_store_for_need"))
        full = locked_pool(100, 1)
        self.assertIsNotNone(full.try_allocate(100, "store", "x", 1))
        occupied = full.occupied()
        self.assertIsNone(full.try_allocate(100, "store", "y", 1))
        self.assertEqual(full.occupied(), occupied)
        names = []
        original = self.pool.reclaim_to_watermark

        def wrapped():
            names.append(threading.current_thread().name)
            return original()

        self.pool.reclaim_to_watermark = wrapped
        self.rt.wait_for_save(0)
        time.sleep(0.15)
        for name in names:
            self.assertEqual(name, "surdata-storeGC")
        pool.close()
        full.close()

    def test_finished_short_tail_is_stored(self):
        gen = self.rt.on_new_request("t", [1] * 20)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("t") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("t", gen, 0, 16, {0: 1, 1: 2}, {0: ev, 1: ev}, token_index=0)
        self.rt.note_block("t", gen, 1, 4, {0: 3, 1: 4}, {0: ev, 1: ev}, token_index=1)
        self.rt.wait_for_save(0)
        self.assertTrue(wait_until(lambda: self.sdk.visible == 16))
        self.rt.request_finished("t", [0, 1], 20)
        self.assertTrue(wait_until(lambda: self.sdk.tails == [4]))
        payload = self.geo.layer_payload(4)
        self.assertTrue(
            any(spans and int(spans[0]["bytes"]) == payload for spans in self.rt.d2h_copies)
        )
        self.assertTrue(wait_until(lambda: self.rt._reqs["t"].blocks[1].d2h_done))
        with self.rt._mu:
            self.assertFalse(any(item[0] == "t" and item[2] == 1 for item in self.rt.drops))
        self.assertEqual(self.geo.dram_layer_offset(1, 4), payload)
        self.assertEqual(self.geo.dram_layer_offset(1, 16), self.geo.layer_payload(16))
        self.sdk.scripted[1] = 20
        self.rt.on_new_request("u", [9] * 20)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("u") == (20, True)))
        self.assertNotEqual(self.sdk.visible, 20)

    def test_mla_4d_geometry(self):
        shape = (25890, 1, 16, 576)
        geo = KvGeometry.from_shape(shape, itemsize=2, num_layers=27)
        self.assertEqual(classify_kv_shape(shape), "mla")
        self.assertEqual(geo.layer_stride(), 16 * 576 * 2)
        self.assertEqual(geo.layer_payload(16), 16 * 576 * 2)
        self.assertEqual(geo.layer_payload(7), 7 * 576 * 2)
        self.assertEqual(geo.kvb_bytes(16), 27 * 16 * 576 * 2)
        self.assertEqual(geo.kvb_bytes(7), 27 * 7 * 576 * 2)
        with self.assertRaises(LayoutError):
            as_mha_axes(shape)
        with self.assertRaises(LayoutError):
            KvGeometry.from_shape((25890, 2, 16, 576), itemsize=2, num_layers=27)

    def test_scheduler_board_and_dedup(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("r", [1] * 32)
        slot = front._pending["r"]["slot"]
        generation = front._pending["r"]["generation"]
        self.assertEqual(front.matched_tokens("r", 0), (0, False))
        self.assertEqual(board.get_at(slot).phase, PHASE_PENDING)
        board.publish_at(slot, generation, PHASE_PRELOADED, 32, 0)
        self.assertEqual(front.matched_tokens("r", 0), (32, True))
        board.publish_at(slot, generation, PHASE_H2D, 32, 0)
        self.assertEqual(front.matched_tokens("r", 0), (None, False))
        board.publish_at(slot, generation, PHASE_MISS, 0, 0)
        self.assertEqual(front.matched_tokens("r", 0), (0, False))
        board.publish_at(slot, 9, PHASE_PRELOADED, 32, 0)
        self.assertEqual(front.matched_tokens("r", 0), (None, False))
        first = front.build_meta()
        second = front.build_meta()
        self.assertEqual(len(first.lookups), 1)
        self.assertEqual(second.lookups, [])
        board.close()

    def test_remember_blocks_keeps_the_full_table(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("r", [1] * 32)
        front.remember_blocks("r", [10, 11], 16, True)
        front.remember_blocks("r", [12], 20, False)
        meta = front.build_meta()
        self.assertEqual(len(meta.save_hints), 2)
        self.assertEqual(meta.save_hints[0]["block_ids"], [10, 11])
        self.assertEqual(meta.save_hints[0]["num_tokens"], 16)
        self.assertEqual(meta.save_hints[1]["block_ids"], [10, 11, 12])
        self.assertEqual(meta.save_hints[1]["num_tokens"], 20)
        front.remember_blocks("r", [], 0, False)
        self.assertEqual(front.build_meta().save_hints, [])
        front.note_finished("r", [10, 11, 12], 20)
        self.assertNotIn("r", front._blocks)
        board.close()

    def test_preloaded_is_answered_before_a_waiting_request_becomes_a_miss(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("wait", [1] * 16)
        front.on_new_request("hit", [2] * 32)
        front.on_new_request("copy", [3] * 32)
        wait_slot = front._pending["wait"]["slot"]
        hit_slot = front._pending["hit"]["slot"]
        copy_slot = front._pending["copy"]["slot"]
        hit_gen = front._pending["hit"]["generation"]
        copy_gen = front._pending["copy"]["generation"]
        board.publish_at(hit_slot, hit_gen, PHASE_PRELOADED, 32, 0)
        board.publish_at(copy_slot, copy_gen, PHASE_H2D, 32, 0)
        self.assertEqual(front.matched_tokens("wait", 0), (None, False))
        self.assertEqual(board.get_at(wait_slot).phase, PHASE_PENDING)
        self.assertEqual(front.matched_tokens("hit", 0), (32, True))
        self.assertEqual(front.matched_tokens("wait", 0), (0, False))
        self.assertEqual(board.get_at(wait_slot).phase, PHASE_PENDING)
        self.assertEqual(front.matched_tokens("copy", 0), (None, False))
        front.build_meta()
        self.assertEqual(front.matched_tokens("wait", 0), (None, False))
        self.assertEqual(front.matched_tokens("hit", 0), (32, True))

    def test_waiting_request_is_a_miss_when_nothing_is_preloaded(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("wait", [1] * 16)
        front.on_new_request("copy", [3] * 32)
        wait_slot = front._pending["wait"]["slot"]
        copy_slot = front._pending["copy"]["slot"]
        copy_gen = front._pending["copy"]["generation"]
        board.publish_at(copy_slot, copy_gen, PHASE_H2D, 32, 0)
        self.assertEqual(front.matched_tokens("wait", 0), (0, False))
        self.assertEqual(board.get_at(wait_slot).phase, PHASE_PENDING)
        self.assertEqual(front.matched_tokens("copy", 0), (None, False))

    def test_miss_and_hbm_are_not_held_for_another_preloaded_request(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("hit", [1] * 32)
        front.on_new_request("miss", [2] * 16)
        front.on_new_request("hbm", [3] * 16)
        hit_slot = front._pending["hit"]["slot"]
        miss_slot = front._pending["miss"]["slot"]
        hbm_slot = front._pending["hbm"]["slot"]
        board.publish_at(hit_slot, front._pending["hit"]["generation"], PHASE_PRELOADED, 32, 0)
        board.publish_at(miss_slot, front._pending["miss"]["generation"], PHASE_MISS, 0, 0)
        board.publish_at(hbm_slot, front._pending["hbm"]["generation"], PHASE_HBM, 16, 0)
        self.assertEqual(front.matched_tokens("miss", 0), (0, False))
        self.assertEqual(front.matched_tokens("hbm", 0), (0, False))
        self.assertEqual(board.get_at(miss_slot).phase, PHASE_MISS)
        self.assertEqual(front.matched_tokens("hit", 0), (32, True))

    def test_mmap_board_roundtrip(self):
        fd, path = tempfile.mkstemp(prefix="surdata-status-")
        os.close(fd)
        writer = StatusBoard(path, create=True)
        writer.publish_at(0, 3, PHASE_PRELOADED, 32, 0)
        reader = StatusBoard(path, create=False)
        view = reader.get_at(0)
        self.assertIsNotNone(view)
        self.assertEqual(view.generation, 3)
        self.assertEqual(view.hit_tokens, 32)
        writer.close()
        reader.close()
        os.remove(path)

    def test_connector_source_has_no_lmcache_import(self):
        path = os.path.join(os.path.dirname(__file__), "..", "surdata_vllm", "connector.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read().lower()
        self.assertNotIn("import lmcache", text)
        self.assertNotIn("lmcacheconnector", text)

    def test_store_full_policy(self):
        self.assertEqual(load_store_full_policy("finish_drop"), "finish_drop")
        self.assertEqual(load_store_full_policy("DROP"), "drop")
        with self.assertRaises(ConfigError):
            load_store_full_policy("defer")
        with self.assertRaises(ConfigError):
            load_store_full_policy("nope")
        settings = load_settings(
            {
                "block_tokens": 16,
                "bytes_per_token": 4,
                "layout_signature": "mla",
            }
        )
        self.assertEqual(settings["store_full_policy"], "finish_drop")
        missing = os.environ.get("SURDATA_SDK_LIB")
        os.environ["SURDATA_SDK_LIB"] = "/no/such/libsurdata_sdk.so"
        try:
            self.assertEqual(load_store_full_policy(), "finish_drop")
            self.assertEqual(load_direct_phase_diag_ms(), 0)
            self.assertEqual(load_direct_save_mode(), "per_layer")
        finally:
            if missing is None:
                os.environ.pop("SURDATA_SDK_LIB", None)
            else:
                os.environ["SURDATA_SDK_LIB"] = missing
        self.assertEqual(load_direct_phase_diag_ms(0), 0)
        self.assertEqual(load_direct_phase_diag_ms(1000), 1000)
        self.assertEqual(load_direct_phase_diag_ms(-5), 0)
        self.assertEqual(load_direct_save_mode("per_layer"), "per_layer")
        self.assertEqual(load_direct_save_mode("END"), "end")
        with self.assertRaises(ConfigError):
            load_direct_save_mode("unmounted")
        with self.assertRaises(ConfigError):
            load_direct_save_mode("off")
        with self.assertRaises(ValueError):
            WorkerRuntime(sdk=object(), policy="finish_drop", save_mode="unmounted")

    def test_phase_diag_snapshot_is_one_line(self):
        rt = WorkerRuntime(sdk=object(), policy="finish_drop")
        rt.queue.enqueue_lookup("r1", {"tokens": [1]}, prompt_tokens=16)
        line = rt.phase_diag_snapshot()
        self.assertTrue(line.startswith("SURDATA_PHASE wait=1 claimed=0 "))
        self.assertIn("publish_wait=0", line)
        self.assertIn("init=0", line)
        self.assertIn("reading=0", line)
        self.assertIn("dram_free=-1", line)
        self.assertIn("order=0", line)
        self.assertIn("in_lookup=0", line)
        self.assertIn("busy=0", line)
        self.assertNotIn("\n", line)
        item = rt.queue.take_next_lookup()
        self.assertIsNotNone(item)
        self.assertTrue(rt.queue.requeue_lookup("r1", item[1], 0))
        parked = rt.phase_diag_snapshot()
        self.assertIn("wait=0", parked)
        self.assertIn("order=1", parked)
        self.assertIn("claimed=0", parked)
        rt._lookup_busy = 1
        self.assertIn("busy=1", rt.phase_diag_snapshot())

    def test_phase_diag_loop_survives_snapshot_error(self):
        rt = WorkerRuntime(sdk=object(), policy="finish_drop")
        rt._phase_diag_ms = 1
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("bad snapshot")
            rt._stop.set()
            return "SURDATA_PHASE ok"

        rt.phase_diag_snapshot = boom
        rt._phase_diag_loop()
        self.assertGreaterEqual(calls["n"], 2)

    def test_sdk_abi_sizes(self):
        sizes = abi_sizes()
        self.assertEqual(sizes, {"init": 48, "hit": 40, "slice": 16, "write": 144, "read": 64})

    def test_same_process_board(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        sched = open_scheduler_board()
        worker = open_worker_board(wait_s=0)
        try:
            front = SchedulerFront(sched)
            front.on_new_request("r", [1] * 32)
            slot = front._pending["r"]["slot"]
            worker.publish_at(slot, 1, PHASE_PRELOADED, 32, 0)
            self.assertEqual(front.matched_tokens("r", 0), (32, True))
            worker.publish_at(slot, 1, PHASE_H2D, 32, 0)
            self.assertEqual(front.matched_tokens("r", 0), (None, False))
            self.assertEqual(sched._path, worker._path)
        finally:
            sched.close()
            worker.close()
            if os.path.exists(path):
                os.remove(path)

    def test_worker_creates_the_board_before_the_scheduler(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        worker = open_worker_board(wait_s=0)
        sched = open_scheduler_board()
        try:
            front = SchedulerFront(sched)
            front.on_new_request("r", [1] * 32)
            slot = front._pending["r"]["slot"]
            worker.publish_at(slot, 1, PHASE_PRELOADED, 32, 0)
            self.assertEqual(front.matched_tokens("r", 0), (32, True))
            self.assertEqual(sched._path, worker._path)
        finally:
            sched.close()
            worker.close()
            if os.path.exists(path):
                os.remove(path)

    def test_mmap_two_writers_keep_every_miss(self):
        fd, path = tempfile.mkstemp(prefix="surdata-status-")
        os.close(fd)
        pending_board = StatusBoard(path, create=True)
        miss_board = StatusBoard(path, create=False)
        front = SchedulerFront(pending_board)
        ids = ["r%d" % i for i in range(800)]
        slots = []
        for req in ids:
            front.on_new_request(req, [1, 2, 3, 4])
            slots.append(front._pending[req]["slot"])
        start = threading.Barrier(2)

        def write_pending():
            start.wait()
            for slot in slots:
                pending_board.publish_at(slot, 1, PHASE_PENDING, 0, 0)

        def write_miss():
            start.wait()
            for slot in slots:
                miss_board.publish_at(slot, 1, PHASE_MISS, 0, 0)

        pending_thread = threading.Thread(target=write_pending)
        miss_thread = threading.Thread(target=write_miss)
        try:
            pending_thread.start()
            miss_thread.start()
            pending_thread.join()
            miss_thread.join()
            missing = [req for req in ids if pending_board.get_at(front._pending[req]["slot"]) is None]
            self.assertEqual(missing, [])
            for req in ids:
                miss_board.publish_at(front._pending[req]["slot"], 1, PHASE_MISS, 0, 0)
            for req in ids:
                self.assertEqual(front.matched_tokens(req, 0), (0, False))
        finally:
            pending_board.close()
            miss_board.close()
            os.remove(path)

    def test_old_generation_does_not_cover_newer_phase(self):
        fd, path = tempfile.mkstemp(prefix="surdata-status-")
        os.close(fd)
        board = StatusBoard(path, create=True)
        try:
            board.publish_at(0, 2, PHASE_PENDING, 0, 0)
            board.publish_at(0, 1, PHASE_MISS, 0, 0)
            view = board.get_at(0)
            self.assertIsNotNone(view)
            self.assertEqual(view.generation, 1)
            self.assertEqual(view.phase, PHASE_MISS)
            board.publish_at(0, 2, PHASE_MISS, 0, 0)
            board.publish_at(0, 1, PHASE_PRELOADED, 32, 0)
            view = board.get_at(0)
            self.assertEqual(view.generation, 1)
            self.assertEqual(view.phase, PHASE_PRELOADED)
            board.publish_at(0, 3, PHASE_CLOSED, 0, 1)
            board.publish_at(0, 2, PHASE_MISS, 0, 0)
            view = board.get_at(0)
            self.assertEqual(view.generation, 2)
            self.assertEqual(view.phase, PHASE_MISS)
        finally:
            board.close()
            os.remove(path)

    def test_miss_reaches_scheduler_when_reqs_missing(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        worker = open_worker_board(wait_s=0)
        sched = open_scheduler_board()
        rt = None
        try:
            self.assertEqual(worker._path, sched._path)
            front = SchedulerFront(sched)
            sdk = FakeSdk()
            rt = WorkerRuntime(sdk=sdk, native=FakeNative(), policy="finish_drop", board=worker)
            rt.start()

            def lookup_hit(req_id, tokens, block_tokens, generation):
                del tokens, block_tokens, generation
                rt._reqs.pop(str(req_id), None)
                if str(req_id) == "gone":
                    return 0
                return 32

            sdk.lookup_hit = lookup_hit
            front.on_new_request("gone", [1] * 16)
            front.on_new_request("orphan-hit", [2] * 32)
            gone_slot = front._pending["gone"]["slot"]
            orphan_slot = front._pending["orphan-hit"]["slot"]
            self.assertEqual(front.matched_tokens("gone", 0), (0, False))
            self.assertEqual(front.matched_tokens("orphan-hit", 0), (0, False))
            self.assertEqual(sched.get_at(gone_slot).phase, PHASE_PENDING)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: sched.get_at(gone_slot).phase == PHASE_MISS))
            self.assertTrue(wait_until(lambda: sched.get_at(orphan_slot).phase == PHASE_MISS))
            gone = sched.get_at(gone_slot)
            orphan = sched.get_at(orphan_slot)
            self.assertEqual(gone.phase, PHASE_MISS)
            self.assertEqual(orphan.phase, PHASE_MISS)
        finally:
            if rt is not None:
                rt.shutdown()
            worker.close()
            sched.close()
            if os.path.exists(path):
                os.remove(path)

    def test_lookup_stays_pending_while_an_earlier_prefix_can_still_publish(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        worker = open_worker_board(wait_s=0)
        sched = open_scheduler_board()
        rt = None
        try:
            front = SchedulerFront(sched)
            sdk = FakeSdk()
            state = {"wait": True}
            calls = []

            def lookup_hit(req_id, tokens, block_tokens, generation):
                del req_id, tokens, block_tokens
                calls.append(int(generation))
                if state["wait"]:
                    return 0
                return 32

            sdk.lookup_hit = lookup_hit
            sdk.prefix_may_arrive = lambda req_id, tokens: state["wait"]
            rt = WorkerRuntime(sdk=sdk, native=FakeNative(), policy="finish_drop", board=worker)
            rt.start()
            front.on_new_request("t2", [1] * 32)
            self.assertEqual(front.matched_tokens("t2", 0), (0, False))
            self.assertEqual(sched.get_at(front._pending["t2"]["slot"]).phase, PHASE_PENDING)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: len(calls) >= 2))
            view = sched.get_at(front._pending["t2"]["slot"])
            self.assertEqual(view.phase, PHASE_PENDING)
            state["wait"] = False
            slot = front._pending["t2"]["slot"]
            self.assertTrue(wait_until(lambda: sched.get_at(slot).phase == PHASE_INIT))
            self.assertNotEqual(sched.get_at(slot).phase, PHASE_MISS)
        finally:
            if rt is not None:
                rt.shutdown()
            worker.close()
            sched.close()
            if os.path.exists(path):
                os.remove(path)

    def test_publish_wait_expires_into_a_miss(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        worker = open_worker_board(wait_s=0)
        sched = open_scheduler_board()
        rt = None
        try:
            front = SchedulerFront(sched)
            sdk = FakeSdk()
            calls = []

            def lookup_hit(req_id, tokens, block_tokens, generation):
                del req_id, tokens, block_tokens, generation
                calls.append(1)
                return 0

            sdk.lookup_hit = lookup_hit
            sdk.prefix_may_arrive = lambda req_id, tokens: True
            rt = WorkerRuntime(
                sdk=sdk,
                native=FakeNative(),
                policy="finish_drop",
                board=worker,
                publish_wait_s=0.2,
            )
            rt.start()
            front.on_new_request("t2", [1] * 32)
            rt.ingest(front.build_meta())
            slot = front._pending["t2"]["slot"]
            self.assertTrue(wait_until(lambda: sched.get_at(slot).phase == PHASE_MISS, timeout=2.0))
            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(front.matched_tokens("t2", 0), (0, False))
        finally:
            if rt is not None:
                rt.shutdown()
            worker.close()
            sched.close()
            if os.path.exists(path):
                os.remove(path)

    def test_own_stale_boot_is_cleared(self):
        path = status_board_path(os.getpid())
        backup = path + ".bak-own-boot"
        had = os.path.exists(path)
        if had:
            shutil.copy2(path, backup)
        board = None
        try:
            plant_board(path, 0, 3, 4, PHASE_PENDING, 8)
            board = open_worker_board(wait_s=0)
            self.assertIsNone(board.get_at(3))
            self.assertEqual(read_boot(path), _process_boot_id())
        finally:
            if board is not None:
                board.close()
            if os.path.exists(path):
                os.remove(path)
            if had:
                shutil.copy2(backup, path)
                os.remove(backup)

    def test_parent_board_boot_mismatch_is_not_truncated(self):
        parent = status_board_path(os.getppid())
        own = status_board_path(os.getpid())
        parent_backup = parent + ".bak-parent-boot"
        own_backup = own + ".bak-own-boot"
        had_parent = os.path.exists(parent)
        had_own = os.path.exists(own)
        if had_parent:
            shutil.copy2(parent, parent_backup)
        if had_own:
            shutil.copy2(own, own_backup)
            os.remove(own)
        board = None
        try:
            plant_board(parent, 0x11111111, 2, 6, PHASE_MISS, 0)
            with open(parent, "rb") as handle:
                before = handle.read()
            board = open_worker_board(wait_s=0)
            with open(parent, "rb") as handle:
                after = handle.read()
            self.assertEqual(before, after)
            view = board.get_at(2)
            self.assertIsNotNone(view)
            self.assertEqual(view.generation, 6)
            self.assertEqual(view.phase, PHASE_MISS)
        finally:
            if board is not None:
                board.close()
            if had_parent:
                shutil.copy2(parent_backup, parent)
                os.remove(parent_backup)
            elif os.path.exists(parent):
                os.remove(parent)
            if had_own:
                if os.path.exists(own):
                    os.remove(own)
                shutil.copy2(own_backup, own)
                os.remove(own_backup)
            elif os.path.exists(own):
                os.remove(own)

    def test_release_reuses_the_same_slot(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        front.on_new_request("a", [1] * 8)
        slot = front._pending["a"]["slot"]
        old = front._pending["a"]["generation"]
        front.note_finished("a", [1], 8)
        self.assertIsNone(board.get_at(slot))
        front.on_new_request("b", [2] * 8)
        self.assertEqual(front._pending["b"]["slot"], slot)
        new = front._pending["b"]["generation"]
        self.assertGreater(new, old)
        board.publish_at(slot, old, PHASE_MISS, 0, 0)
        view = board.get_at(slot)
        self.assertEqual(view.generation, old)
        self.assertEqual(view.phase, PHASE_MISS)
        board.release(slot, old)
        self.assertIsNone(board.get_at(slot))

    def test_free_list_empty_does_not_publish(self):
        board = StatusBoard()
        front = SchedulerFront(board)
        for i in range(_SLOTS):
            front.on_new_request("r%d" % i, [1])
        self.assertEqual(front.on_new_request("overflow", [1]), 0)
        self.assertEqual(front.matched_tokens("overflow", 0), (None, False))
        self.assertNotIn("overflow", front._pending)

    def test_lookup_error_stays_pending(self):
        self.sdk.fail_lookup = True
        self.rt.on_new_request("e", [1] * 16)
        self.assertTrue(
            wait_until(
                lambda: self.rt.queue.get("e") is not None
                and self.rt.queue.get("e").status == STATUS_FALLBACK_MISS
            )
        )

    def test_preload_read_error_is_a_miss(self):
        self.sdk.fail_read = True
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("pf", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("pf") == (0, False)))
        self.assertEqual(self.rt.queue.get("pf").status, STATUS_FALLBACK_MISS)
        self.assertEqual(self.rt.queue.get("pf").lookup_hit_tokens, 0)

    def test_preload_callback_error_is_a_miss(self):
        self.sdk.read_status = 3
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("cb", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("cb") == (0, False)))
        self.assertEqual(self.rt.queue.get("cb").status, STATUS_FALLBACK_MISS)

    def test_read_failure_after_preload_is_a_miss(self):
        self.sdk.scripted[1] = 16
        gen = self.rt.on_new_request("late", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("late") == (16, True)))
        with self.rt._mu:
            slots = getattr(self.rt._reqs["late"], "_preload_slots", None)
        self.rt._on_preloaded("late", gen, 16, slots, 3)
        self.assertEqual(self.rt.matched_tokens("late"), (0, False))
        self.assertEqual(self.rt.queue.get("late").status, STATUS_FALLBACK_MISS)
        self.assertEqual(self.rt.queue.get("late").lookup_hit_tokens, 0)

    def test_mark_reading_failure_is_retried(self):
        calls = {"n": 0}
        real = self.rt.queue.mark_reading

        def wrapped(req_id, generation):
            calls["n"] += 1
            if calls["n"] == 1:
                return False
            return real(req_id, generation)

        self.rt.queue.mark_reading = wrapped
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("r", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("r") == (16, True)))
        self.assertGreaterEqual(calls["n"], 2)

    def test_h2d_phase_does_not_allocate_again(self):
        self.native.hold = True
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("z", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("z") == (16, True)))
        self.rt.observe_layer("0", 1000, None)
        self.rt.observe_layer("1", 2000, None)
        self.rt.update_state_after_alloc("z", [1], 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("z") == (None, False)))

    def test_watermark_covers_snapshot_and_next_block(self):
        full = self.geo.kvb_bytes(16)
        self.store_pool.watermark_bytes = full
        gen = self.rt.on_new_request("a", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("a") == (0, False)))
        ev = self.native.event_create()
        self.native.event_record(ev)
        self.rt.note_block("a", gen, 1, 16, {0: 5}, {0: ev})
        self.rt.wait_for_save()
        level, span = self.store_pool.published_watermark()
        self.assertEqual(level, full)
        self.assertEqual(span, full)

    def test_zero_need_keeps_minimum_watermark(self):
        full = self.geo.kvb_bytes(16)
        self.store_pool.watermark_bytes = full
        for index in range(self.store_pool.slot_count):
            taken = self.store_pool.try_allocate(full, "store", "old%d" % index, 1)
            self.assertIsNotNone(taken)
            self.store_pool.mark_stored(taken)
            self.store_pool.unpin(taken)
        self.assertEqual(self.store_pool.ledger_free(), 0)
        self.rt.on_new_request("idle", [1] * 16)
        self.assertTrue(wait_until(lambda: self.rt.matched_tokens("idle") == (0, False)))
        self.assertTrue(self.rt.request_finished("idle", [3], 16))
        self.rt.shutdown()
        self.rt.wait_for_save()
        level, span = self.store_pool.published_watermark()
        self.assertEqual(level, full)
        self.assertEqual(span, 0)
        self.assertEqual(self.store_pool.reclaim_to_watermark(), full)
        self.assertEqual(self.store_pool.ledger_free(), full)

    def test_preload_free_wakes_preload_store_free_does_not(self):
        self.rt.shutdown()
        self.preload_pool.close()
        self.store_pool.close()
        slot = self.geo.kvb_bytes(16)
        self.preload_pool = locked_pool(slot, 1, kind="preload")
        self.store_pool = locked_pool(slot, 1, kind="store")
        self.pool = self.store_pool
        self.rt = WorkerRuntime(
            sdk=self.sdk,
            native=self.native,
            policy="finish_drop",
            preload_pool=self.preload_pool,
            store_pool=self.store_pool,
        )
        self.rt.set_geometry(self.geo)
        self.rt.start()
        held = self.preload_pool.try_allocate(slot, "preload", "x", 1)
        stored = self.store_pool.try_allocate(slot, "store", "y", 1)
        self.store_pool.mark_stored(stored)
        self.store_pool.unpin(stored)
        self.sdk.scripted[1] = 16
        self.rt.on_new_request("w", [1] * 16)
        time.sleep(0.2)
        self.assertEqual(self.sdk.reads, [])
        self.store_pool.request_watermark(slot, slot)
        self.assertTrue(wait_until(lambda: self.store_pool.occupied() == 0))
        time.sleep(0.15)
        self.assertEqual(self.sdk.reads, [])
        self.preload_pool.unpin(held)
        self.assertTrue(wait_until(lambda: len(self.sdk.reads) >= 1))

    def test_try_allocate_does_not_reclaim(self):
        pool = locked_pool(100, 1, kind="store")
        self.assertIsNotNone(pool.try_allocate(100, "store", "x", 1))
        pool.reclaim_to_watermark = lambda: (_ for _ in ()).throw(AssertionError("gc in try_allocate"))
        self.assertIsNone(pool.try_allocate(100, "store", "y", 1))
        pool.close()

    def test_prefix_hash_is_only_in_the_sdk(self):
        root = os.path.join(os.path.dirname(__file__), "..", "surdata_vllm")
        bridge = ""
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".py"):
                    continue
                with open(os.path.join(dirpath, name), encoding="utf-8") as handle:
                    text = handle.read()
                lowered = text.lower()
                self.assertNotIn("import hashlib", lowered)
                self.assertNotIn("import xxhash", lowered)
                self.assertNotIn("xxh3", lowered)
                if name == "sdk_bridge.py":
                    bridge = text
        self.assertIn("surdata_sdk_prefix_hashes_from_tokens", bridge)
        self.assertIn("surdata_sdk_lookup_prefix_hit", bridge)
        self.assertIn("surdata_sdk_request_set_prefix_hashes", bridge)
        path = os.path.join(root, "connector.py")
        with open(path, encoding="utf-8") as handle:
            connector = handle.read()
        self.assertIn("def get_transfer_results(self, finished_req_ids=None)", connector)
        self.assertIn("finished_sending", connector)
        self.assertNotIn("import lmcache", connector.lower())


class LookupWindowTest(unittest.TestCase):
    def test_full_window_does_not_take_the_next(self):
        queue = DramPipelineQueue(1)
        free = {"bytes": 1}
        queue.set_capacity(lambda entry: 1, lambda: free["bytes"])
        queue.enqueue_lookup("a", {"tokens": [1]}, 1, generation=1)
        queue.enqueue_lookup("b", {"tokens": [2]}, 1, generation=1)
        self.assertEqual(queue.take_next_lookup()[0], "a")
        free["bytes"] = 0
        self.assertIsNone(queue.take_next_lookup())
        self.assertEqual(queue.get("b").status, STATUS_LOOKUP_PENDING)
        self.assertFalse(queue.get("b").lookup_claimed)
        free["bytes"] = 1
        self.assertEqual(queue.take_next_lookup()[0], "b")

    def test_mark_done_lets_the_fifo_next_lookup(self):
        queue = DramPipelineQueue(1)
        queue.enqueue_lookup("a", {"tokens": [1]}, 16, generation=1)
        queue.enqueue_lookup("b", {"tokens": [2]}, 16, generation=1)
        self.assertEqual(queue.take_next_lookup()[0], "a")
        nxt = queue.take_next_lookup()
        self.assertEqual(nxt[0], "b")
        self.assertTrue(queue.mark_done("a", 1))

    def test_miss_stays_held_until_release(self):
        queue = DramPipelineQueue(1)
        queue.enqueue_lookup("a", {"tokens": [1]}, 16, generation=1)
        queue.enqueue_lookup("b", {"tokens": [2]}, 16, generation=1)
        queue.take_next_lookup()
        self.assertTrue(queue.complete_lookup("a", 1, 0, 16))
        self.assertEqual(queue.get("a").status, STATUS_FALLBACK_MISS)
        self.assertFalse(queue.release_lookup_window("a"))
        self.assertEqual(queue.take_next_lookup()[0], "b")

    def test_close_returns_the_place(self):
        queue = DramPipelineQueue(1)
        queue.enqueue_lookup("a", {"tokens": [1]}, 1, generation=1)
        queue.enqueue_lookup("b", {"tokens": [2]}, 1, generation=1)
        queue.take_next_lookup()
        queue.remove("a")
        self.assertEqual(queue.take_next_lookup()[0], "b")

    def test_release_does_not_mention_durable_commit(self):
        text = "\n".join(
            [
                inspect.getsource(DramPipelineQueue.release_lookup_window),
                inspect.getsource(DramPipelineQueue.mark_done),
                inspect.getsource(DramPipelineQueue.remove),
                inspect.getsource(WorkerRuntime.update_state_after_alloc),
            ]
        )
        self.assertNotIn("kv_durable", text)
        self.assertNotIn("commit_kv", text)
        self.assertNotIn("_lock", inspect.getsource(DramPipelineQueue.__init__))

    def _window(self):
        geo = KvGeometry(100, 16, 4, 2, 2)
        sdk = FakeSdk()
        native = FakeNative()
        slot = geo.kvb_bytes(16)
        preload = locked_pool(slot, 4, kind="preload")
        store = locked_pool(slot, 4, kind="store")
        board = StatusBoard()
        front = SchedulerFront(board)
        rt = WorkerRuntime(
            sdk=sdk,
            native=native,
            policy="finish_drop",
            board=board,
            preload_pool=preload,
            store_pool=store,
            lookup_limit=1,
        )
        rt.set_geometry(geo)
        rt._torch = FakeTorch()
        rt.start()
        return rt, sdk, native, board, front, preload, store

    def _close(self, rt, preload, store, board):
        rt.shutdown()
        preload.close()
        store.close()
        board.close()

    def _watch(self, sdk):
        seen = []
        orig = sdk.lookup_hit

        def wrapped(req_id, tokens, block, generation):
            seen.append(str(req_id))
            return orig(req_id, tokens, block, generation)

        sdk.lookup_hit = wrapped
        return seen

    def test_full_window_leaves_the_board_pending(self):
        rt, sdk, _native, board, front, preload, store = self._window()
        gate = threading.Event()
        sdk.gates[1] = gate
        seen = self._watch(sdk)
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: seen == ["a"]))
            view = board.get_at(front._pending["b"]["slot"])
            self.assertEqual(view.phase, PHASE_PENDING)
            self.assertNotEqual(view.phase, PHASE_MISS)
            self.assertFalse(rt.queue.get("b").lookup_claimed)
        finally:
            gate.set()
            self._close(rt, preload, store, board)

    def test_hbm_then_fifo_next_lookup(self):
        rt, sdk, native, board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        seen = self._watch(sdk)
        started_while_preloaded = []
        orig = sdk.lookup_hit

        def wrapped(req_id, tokens, block, generation):
            if str(req_id) == "b":
                with rt._mu:
                    req = rt._reqs.get("a")
                    phase = None if req is None else req.phase
                    blocks = [] if req is None else list(req.block_ids)
                started_while_preloaded.append(
                    phase == PHASE_PRELOADED and not blocks and "a" not in rt._recving
                )
            return orig(req_id, tokens, block, generation)

        sdk.lookup_hit = wrapped
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertTrue(wait_until(lambda: "b" in seen))
            self.assertEqual(started_while_preloaded, [True])
            self.assertNotIn("a", rt._recving)
            with rt._mu:
                self.assertFalse(rt._reqs["a"].block_ids)
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [0], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with native.lock:
                for kind, seq in native.submits:
                    if kind == "h2d":
                        native.done.append({"sequence": seq, "status": 0, "direction": 1})
            rt._wake_h2d.set()
            self.assertTrue(wait_until(lambda: "a" in rt.transfer_results().finished_recving))
            self.assertIsNone(rt.queue.get("a"))
            view = board.get_at(front._pending["a"]["slot"])
            self.assertEqual(view.phase, PHASE_HBM)
        finally:
            self._close(rt, preload, store, board)

    def test_preloaded_without_alloc_frees_the_lookup_window(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        sdk.scripted[1] = 16
        seen = self._watch(sdk)
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_PRELOADED)
                self.assertFalse(rt._reqs["a"].block_ids)
            self.assertTrue(wait_until(lambda: "b" in seen))
            self.assertEqual(self._h2d_submits(native), [])
        finally:
            self._close(rt, preload, store, front.board)

    def test_miss_note_alloc_returns_the_place(self):
        rt, sdk, _native, board, front, preload, store = self._window()
        sdk.scripted[1] = 0
        commits = []
        sdk.kv_durable_commit = lambda *args, **kwargs: commits.append(1)
        seen = self._watch(sdk)
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (0, False)))
            self.assertTrue(wait_until(lambda: "b" in seen))
            self.assertEqual(rt.queue.get("a").status, STATUS_FALLBACK_MISS)
            self.assertEqual(rt.queue.get("b").status, STATUS_FALLBACK_MISS)
            front.note_alloc("a", [4], 0)
            rt.ingest(front.build_meta())
            self.assertEqual(commits, [])
        finally:
            self._close(rt, preload, store, board)

    def test_late_read_failure_frees_the_lookup_window(self):
        rt, sdk, _native, _board, front, preload, store = self._window()
        sdk.scripted[1] = 16
        seen = self._watch(sdk)
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertTrue(wait_until(lambda: "b" in seen))
            with rt._mu:
                req = rt._reqs["a"]
                slots = getattr(req, "_preload_slots", None)
                gen = req.generation
            rt._on_preloaded("a", gen, 16, slots, 3)
            self.assertEqual(rt.matched_tokens("a"), (0, False))
            self.assertTrue(wait_until(lambda: "b" in seen))
        finally:
            self._close(rt, preload, store, front.board)

    def _h2d_submits(self, native):
        with native.lock:
            return [item for item in native.submits if item[0] == "h2d"]

    def test_preload_then_alloc_arms_h2d_once(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_PRELOADED)
                self.assertFalse(rt._reqs["a"].block_ids)
            self.assertEqual(self._h2d_submits(native), [])
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
            self.assertEqual(len(self._h2d_submits(native)), 1)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_alloc_then_preload_arms_h2d(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        gate = threading.Event()
        sdk.read_gate = gate
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: len(sdk.reads) == 1))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_READING)
            self.assertEqual(self._h2d_submits(native), [])
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [4], 16))
            self.assertEqual(self._h2d_submits(native), [])
            gate.set()
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            gate.set()
            self._close(rt, preload, store, front.board)

    def test_zero_external_tokens_does_not_submit_h2d(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertEqual(self._h2d_submits(native), [])
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            with rt._mu:
                held = list(rt._reqs["a"]._preload_slots)
            self.assertFalse(rt.update_state_after_alloc("a", [3], 0))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_FALLBACK_MISS))
            self.assertEqual(len(self._h2d_submits(native)), 0)
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_MISS)
                generation = int(rt._reqs["a"].generation)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_MISS)
            got = rt.transfer_results()
            self.assertNotIn("a", got.finished_recving)
            self.assertNotIn("a", got.failed_recving)
            self.assertTrue(all(slot.freed for slot in held))
            rt.ingest_blocks("a", generation, [3], 16)
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 0)
            got = rt.transfer_results()
            self.assertNotIn("a", got.finished_recving)
            self.assertNotIn("a", got.failed_recving)
        finally:
            self._close(rt, preload, store, front.board)

    def test_alloc_before_request_is_applied_at_preload(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            self.assertNotIn("a", rt._reqs)
            self.assertFalse(rt.update_state_after_alloc("a", [9], 16))
            rt.set_layer_bases({"0": 1000, "1": 2000})
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.queue.get("a") is not None and rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].block_ids, [9])
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_ingest_blocks_and_alloc_submit_h2d_once(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [5], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with rt._mu:
                generation = int(rt._reqs["a"].generation)
            rt.ingest_blocks("a", generation, [5], 16)
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_missing_block_ids_is_not_submitted(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_PRELOADED)
                self.assertTrue(rt._reqs["a"]._preload_slots)
                self.assertFalse(rt._reqs["a"].block_ids)
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(self._h2d_submits(native), [])
            self.assertGreaterEqual(rt.h2d_skip_empty, 1)
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_PRELOADED)
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [7], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_queue_full_stays_preloaded_then_submits(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        native.h2d_rcs = [1, 1, 1, 1, 1]
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            with rt._mu:
                held = list(rt._reqs["a"]._preload_slots)
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            time.sleep(0.15)
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
            self.assertGreaterEqual(rt.h2d_skip_full, 1)
            self.assertNotIn("a", rt.transfer_results().failed_recving)
            self.assertFalse(any(slot.freed for slot in held))
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_PRELOADED)
            native.h2d_rcs[:] = [0]
            rt._wake_h2d.set()
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
            self.assertFalse(any(slot.freed for slot in held))
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_short_layer_bases_stays_preloaded(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: rt.h2d_skip_geometry >= 1))
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
            self.assertEqual(self._h2d_submits(native), [])
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_PRELOADED)
        finally:
            self._close(rt, preload, store, front.board)

    def test_two_preloaded_requests_both_enter_h2d_before_poll(self):
        rt, sdk, native, board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertTrue(wait_until(lambda: rt.matched_tokens("b") == (16, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(rt.update_state_after_alloc("b", [4], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertTrue(wait_until(lambda: rt.queue.get("b").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
            self.assertEqual(board.get_at(front._pending["b"]["slot"]).phase, PHASE_H2D)
            self.assertEqual(len(self._h2d_submits(native)), 2)
            self.assertNotIn("a", rt.transfer_results().finished_recving)
            self.assertNotIn("b", rt.transfer_results().finished_recving)
            with native.lock:
                for kind, seq in native.submits:
                    if kind == "h2d":
                        native.done.append({"sequence": seq, "status": 0, "direction": 1})
            rt._wake_h2d.set()
            self.assertTrue(wait_until(lambda: "a" in rt.transfer_results().finished_recving and "b" in rt.transfer_results().finished_recving))
            self.assertIsNone(rt.queue.get("a"))
            self.assertIsNone(rt.queue.get("b"))
            self.assertEqual(board.get_at(front._pending["a"]["slot"]).phase, PHASE_HBM)
            self.assertEqual(board.get_at(front._pending["b"]["slot"]).phase, PHASE_HBM)
        finally:
            self._close(rt, preload, store, board)

    def test_h2d_invalid_submit_is_a_miss_and_unpins(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.h2d_rcs = [2]
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            with rt._mu:
                held = list(rt._reqs["a"]._preload_slots)
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_FALLBACK_MISS))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_MISS)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_MISS)
            self.assertIn("a", rt.transfer_results().failed_recving)
            self.assertEqual(len(self._h2d_submits(native)), 1)
            self.assertTrue(all(slot.freed for slot in held))
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_contiguous_blocks_are_one_span_per_layer(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 32)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [7, 8], 32))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(len(self._h2d_submits(native)), 1)
            spans = native.h2d_spans[-1]
            self.assertEqual(len(spans), 2)
            self.assertEqual({span["bytes"] for span in spans}, {256})
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_gap_in_block_ids_splits_each_layer(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 32)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [7, 9], 32))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(len(self._h2d_submits(native)), 1)
            spans = native.h2d_spans[-1]
            self.assertEqual(len(spans), 4)
            self.assertTrue(all(span["bytes"] == 128 for span in spans))
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_uses_hit_prefix_when_block_table_is_longer(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 128)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [7, 8, 9, 10, 11, 12, 13, 14], 32))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(len(self._h2d_submits(native)), 1)
            spans = native.h2d_spans[-1]
            self.assertEqual(len(spans), 2)
            payload = rt.geometry.layer_payload(16)
            stride = rt.geometry.layer_stride()
            with rt._mu:
                slots = list(rt._reqs["a"]._preload_slots)
            self.assertEqual({span["bytes"] for span in spans}, {payload * 2})
            for span in spans:
                layer = int(span["layer"])
                self.assertEqual(span["src"], slot_addr(slots, rt.geometry.dram_prefix_offset(layer, 0, 2)))
                base = 1000 if layer == 0 else 2000
                self.assertEqual(span["dst"], base + 7 * stride)
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_waits_for_full_table_when_alloc_is_not_the_prefix(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 128)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [100, 101, 102, 103], 16))
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 0)
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
            with rt._mu:
                generation = int(rt._reqs["a"].generation)
            rt.ingest_blocks("a", generation, [7, 8, 9, 10], 64)
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            self.assertEqual(len(self._h2d_submits(native)), 1)
            spans = native.h2d_spans[-1]
            stride = rt.geometry.layer_stride()
            self.assertEqual(spans[0]["dst"], 1000 + 7 * stride)
            self.assertEqual(spans[1]["dst"], 2000 + 7 * stride)
            self.assertTrue(all(span["bytes"] == rt.geometry.layer_payload(16) * 2 for span in spans))
        finally:
            self._close(rt, preload, store, front.board)

    def test_short_full_block_table_warns_and_misses(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 128)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            with rt._mu:
                generation = int(rt._reqs["a"].generation)
                held = list(rt._reqs["a"]._preload_slots)
            with self.assertLogs("surdata_vllm.streams", level="WARNING") as caught:
                rt.ingest_blocks("a", generation, [7], 32)
                self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_FALLBACK_MISS))
            text = "\n".join(caught.output)
            self.assertIn("req=a", text)
            self.assertIn("blocks=1", text)
            self.assertIn("hit_blocks=2", text)
            self.assertIn("hit_tokens=32", text)
            self.assertIn("generation=%s" % generation, text)
            self.assertEqual(len(self._h2d_submits(native)), 0)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_MISS)
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_MISS)
                self.assertIsNone(rt._reqs["a"]._preload_slots)
            self.assertIn("a", rt.transfer_results().failed_recving)
            self.assertTrue(all(slot.freed for slot in held))
        finally:
            self._close(rt, preload, store, front.board)

    def test_short_alloc_without_full_table_stays_preloaded(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 128)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [7], 16))
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 0)
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_PRELOADED)
            with rt._mu:
                self.assertFalse(rt._reqs["a"].full_block_table)
                self.assertEqual(rt._reqs["a"].phase, PHASE_PRELOADED)
                self.assertIsNotNone(rt._reqs["a"]._preload_slots)
        finally:
            self._close(rt, preload, store, front.board)

    def test_h2d_span_error_misses_one_request_and_continues(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        import surdata_vllm.streams as streams_mod

        original = streams_mod.slot_addr
        calls = {"n": 0}

        def boom(slots, offset):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("KVB offset is outside the pinned slots")
            return original(slots, offset)

        try:
            front.on_new_request("a", [1] * 16)
            front.on_new_request("b", [2] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            self.assertTrue(wait_until(lambda: rt.matched_tokens("b") == (16, True)))
            streams_mod.slot_addr = boom
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(rt.update_state_after_alloc("b", [4], 16))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_FALLBACK_MISS))
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_MISS)
            self.assertIn("a", rt.transfer_results().failed_recving)
            self.assertTrue(wait_until(lambda: len(self._h2d_submits(native)) == 1))
            self.assertEqual(rt.queue.get("b").status, STATUS_H2D_IN_FLIGHT)
            self.assertEqual(front.board.get_at(front._pending["b"]["slot"]).phase, PHASE_H2D)
        finally:
            streams_mod.slot_addr = original
            self._close(rt, preload, store, front.board)

    def test_mark_h2d_false_does_not_submit_again(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: len(self._h2d_submits(native)) == 1))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_H2D_IN_FLIGHT))
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_cancel_during_submit_cancels_that_sequence(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16

        def cancel_first():
            rt.cancel("a")

        native.h2d_hook = cancel_first
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            slot = front._pending["a"]["slot"]
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: front.board.get_at(slot).phase == PHASE_CLOSED))
            self.assertEqual(len(self._h2d_submits(native)), 1)
            sequence = int(self._h2d_submits(native)[0][1])
            self.assertIn(sequence, native.cancelled)
            self.assertEqual(front.board.get_at(slot).phase, PHASE_CLOSED)
            with native.lock:
                native.done.append({"sequence": sequence, "status": 1, "direction": 1})
            rt._poll_copies()
            self.assertNotIn("a", rt.transfer_results().finished_recving)
            self.assertEqual(front.board.get_at(slot).phase, PHASE_CLOSED)
        finally:
            self._close(rt, preload, store, front.board)

    def test_publish_retry_does_not_mark_miss(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        h2d_publishes = []
        orig = rt._publish

        def wrapped(req_id, generation, phase, hit, closed=0):
            if int(phase) == PHASE_H2D:
                h2d_publishes.append(int(phase))
                if len(h2d_publishes) == 1:
                    raise RuntimeError("board")
            return orig(req_id, generation, phase, hit, closed)

        rt._publish = wrapped
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: len(self._h2d_submits(native)) == 1))
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_H2D)
            self.assertEqual(front.board.get_at(front._pending["a"]["slot"]).phase, PHASE_H2D)
            self.assertNotIn("a", rt.transfer_results().failed_recving)
            self.assertNotEqual(rt.queue.get("a").status, STATUS_FALLBACK_MISS)
            self.assertEqual(len(h2d_publishes), 2)
            rt._wake_h2d.set()
            time.sleep(0.15)
            self.assertEqual(len(self._h2d_submits(native)), 1)
            self.assertEqual(len(h2d_publishes), 2)
        finally:
            self._close(rt, preload, store, front.board)

    def test_mark_h2d_raise_still_reaches_hbm(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16

        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: len(self._h2d_submits(native)) == 1))
            self.assertNotIn("a", rt.transfer_results().failed_recving)
            self.assertNotEqual(rt.queue.get("a").status, STATUS_FALLBACK_MISS)
            with native.lock:
                native.done.append(
                    {"sequence": native.submits[-1][1], "status": 0, "direction": 1}
                )
            rt._poll_copies()
            with rt._mu:
                self.assertEqual(rt._reqs["a"].phase, PHASE_HBM)
            self.assertIsNone(rt.queue.get("a"))
            self.assertIn("a", rt.transfer_results().finished_recving)
        finally:
            self._close(rt, preload, store, front.board)

    def test_cancel_before_accept_publish_stays_closed(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.hold = True
        sdk.scripted[1] = 16
        h2d_publishes = []
        orig = rt._publish

        def wrapped(req_id, generation, phase, hit, closed=0):
            if int(phase) == PHASE_H2D:
                h2d_publishes.append(int(phase))
            return orig(req_id, generation, phase, hit, closed)

        rt._publish = wrapped

        def cancel_first():
            rt.cancel("a")

        native.h2d_hook = cancel_first
        try:
            front.on_new_request("a", [1] * 16)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (16, True)))
            slot = front._pending["a"]["slot"]
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3], 16))
            self.assertTrue(wait_until(lambda: front.board.get_at(slot).phase == PHASE_CLOSED))
            self.assertEqual(h2d_publishes, [PHASE_H2D])
            time.sleep(0.15)
            self.assertEqual(front.board.get_at(slot).phase, PHASE_CLOSED)
            self.assertEqual(h2d_publishes, [PHASE_H2D])
            self.assertEqual(len(self._h2d_submits(native)), 1)
        finally:
            self._close(rt, preload, store, front.board)

    def test_cancel_drops_publish_wait(self):
        rt, _sdk, _native, _board, _front, preload, store = self._window()
        try:
            rt._publish_wait_since[("c", 1)] = 0
            rt._publish_wait_since[("c", 2)] = 0
            rt._publish_wait_since[("other", 1)] = 0
            rt.cancel("c")
            self.assertFalse(any(item[0] == "c" for item in rt._publish_wait_since))
            self.assertIn(("other", 1), rt._publish_wait_since)
        finally:
            self._close(rt, preload, store, _front.board)

    def test_cancel_during_publish_wait_keeps_board_closed(self):
        path = status_board_path(os.getpid())
        if os.path.exists(path):
            os.remove(path)
        worker = open_worker_board(wait_s=0)
        sched = open_scheduler_board()
        rt = None
        release = threading.Event()
        try:
            front = SchedulerFront(sched)
            sdk = FakeSdk()
            started = threading.Event()

            def lookup_hit(req_id, tokens, block_tokens, generation):
                del req_id, tokens, block_tokens, generation
                started.set()
                release.wait(2)
                return 0

            sdk.lookup_hit = lookup_hit
            sdk.prefix_may_arrive = lambda req_id, tokens: True
            rt = WorkerRuntime(sdk=sdk, native=FakeNative(), policy="finish_drop", board=worker)
            rt.start()
            front.on_new_request("t2", [1] * 32)
            slot = front._pending["t2"]["slot"]
            rt.ingest(front.build_meta())
            self.assertTrue(started.wait(2))
            rt.cancel("t2")
            release.set()
            time.sleep(0.3)
            self.assertEqual(sched.get_at(slot).phase, PHASE_CLOSED)
        finally:
            release.set()
            if rt is not None:
                rt.shutdown()
            worker.close()
            sched.close()
            if os.path.exists(path):
                os.remove(path)

    def test_hbm_preempt_parks_preloaded_without_a_lookup_slot(self):
        rt, _sdk, native, board, front, preload, store = self._window()
        rt.shutdown()
        try:
            slot = 0
            front._pending["a"] = {"tokens": [1] * 16, "generation": 1, "slot": slot}
            front._slot_gen[slot] = 1
            board.publish_at(slot, 1, PHASE_HBM, 0, 0)
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_HBM,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._slots["a"] = slot
            front.note_preempt("a")
            self.assertEqual(front._pending["a"]["generation"], 1)
            self.assertEqual(board.get_at(slot).phase, PHASE_HBM)
            rt.preempt("a")
            self.assertEqual(board.get_at(slot).phase, PHASE_READING)
            self.assertNotEqual(board.get_at(slot).phase, PHASE_CLOSED)
            d2h = [item for item in native.submits if item[0] == "d2h"]
            self.assertEqual(len(d2h), 1)
            span = native.spans[-1][0]
            self.assertEqual(span["src"], 1000 + 3 * 128)
            self.assertEqual(span["bytes"], 128)
            rt._poll_copies()
            parked = rt.queue.get("a")
            self.assertEqual(parked.status, STATUS_PRELOADED)
            self.assertEqual(parked.actual_tokens, 16)
            self.assertEqual(board.get_at(slot).phase, PHASE_PRELOADED)
            self.assertEqual(board.get_at(slot).hit_tokens, 16)
            rt.queue.enqueue_lookup("b", {"tokens": [2]}, 16, generation=1)
            self.assertEqual(rt.queue.take_next_lookup()[0], "b")
            with rt._mu:
                self.assertEqual(rt._reqs["a"].block_ids, [])
        finally:
            self._close(rt, preload, store, board)

    def test_preempt_before_hbm_still_closes(self):
        rt, _sdk, _native, board, front, preload, store = self._window()
        rt.shutdown()
        try:
            slot = 1
            front._pending["c"] = {"tokens": [1] * 16, "generation": 1, "slot": slot}
            front._slot_gen[slot] = 1
            board.publish_at(slot, 1, PHASE_PENDING, 0, 0)
            with rt._mu:
                rt._reqs["c"] = _Req(req_id="c", generation=1, tokens=[1] * 16, phase=PHASE_PENDING)
            rt._slots["c"] = slot
            front.note_preempt("c")
            self.assertEqual(front._pending["c"]["generation"], 1)
            self.assertEqual(board.get_at(slot).phase, PHASE_PENDING)
            rt.preempt("c")
            with rt._mu:
                self.assertTrue(rt._reqs["c"].closed)
            self.assertEqual(board.get_at(slot).phase, PHASE_CLOSED)
        finally:
            self._close(rt, preload, store, board)

    def test_park_waits_when_the_preload_pool_is_full(self):
        rt, _sdk, native, board, front, preload, store = self._window()
        rt.shutdown()
        try:
            held = []
            while True:
                got = preload.try_allocate(preload.slot_bytes, "preload", "pad", 1)
                if not got:
                    break
                held.append(got)
            self.assertTrue(held)
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_HBM,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._slots["a"] = 0
            board.publish_at(0, 1, PHASE_HBM, 0, 0)
            rt.preempt("a")
            self.assertEqual(board.get_at(0).phase, PHASE_READING)
            self.assertFalse(any(item[0] == "d2h" for item in native.submits))
            self.assertNotEqual(board.get_at(0).phase, PHASE_CLOSED)
            preload.abort_unstored(held.pop())
            rt._drain_parks()
            self.assertEqual(sum(1 for item in native.submits if item[0] == "d2h"), 1)
        finally:
            self._close(rt, preload, store, board)

    def test_alloc_keeps_the_full_block_table(self):
        rt, _sdk, native, _board, _front, preload, store = self._window()
        rt.shutdown()
        try:
            generation = rt.queue.enqueue_lookup("p", {"tokens": [1] * 32}, 32)
            self.assertTrue(rt.queue.complete_lookup("p", generation, 32, 32))
            self.assertTrue(rt.queue.mark_reading("p", generation))
            self.assertTrue(rt.queue.mark_preloaded("p", generation, 32))
            with rt._mu:
                rt._reqs["p"] = _Req(
                    req_id="p",
                    generation=generation,
                    tokens=[1] * 32,
                    phase=PHASE_PRELOADED,
                    hit_tokens=32,
                    block_ids=[7, 8, 9, 10],
                    full_block_table=True,
                    layer_base={"0": 1000, "1": 5000},
                )
                rt._reqs["p"]._preload_slots = [type("S", (), {"nbytes": 1024, "buf": 0x1000, "freed": False})()]
            rt.ingest_blocks("p", generation, [7, 8, 9, 10], 32)
            rt.update_state_after_alloc("p", [100, 101, 102, 103], 32)
            rt._submit_preloaded_h2d("p", generation)
            self.assertEqual(len(self._h2d_submits(native)), 1)
            self.assertEqual(native.h2d_spans[-1][0]["dst"], 1000 + 7 * 128)
            with rt._mu:
                self.assertEqual(rt._reqs["p"].block_ids, [7, 8, 9, 10])
                self.assertTrue(rt._reqs["p"].full_block_table)
                self.assertEqual(rt._reqs["p"].external_tokens, 32)
        finally:
            self._close(rt, preload, store, _front.board)

    def test_finished_park_job_moves_into_the_store_pool(self):
        rt, sdk, native, board, _front, preload, store = self._window()
        rt.shutdown()
        try:
            native.poll_hold = True
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_HBM,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._slots["a"] = 0
            board.publish_at(0, 1, PHASE_HBM, 0, 0)
            rt.preempt("a")
            self.assertEqual(sum(1 for item in native.submits if item[0] == "d2h"), 1)
            self.assertTrue(rt._park_in_jobs("a"))
            rt.request_finished("a", [3], 16)
            native.poll_hold = False
            rt._poll_copies()
            self.assertIsNone(rt.queue.get("a"))
            self.assertTrue(sdk.save_slots)
            self.assertTrue(all(slot.purpose == "store" for slot in sdk.save_slots[-1]))
            self.assertFalse(any(not slot.freed and slot.purpose == "preload" and slot.req_id == "a" for slot in preload._slots))
        finally:
            self._close(rt, preload, store, board)

    def test_finished_while_park_is_waiting_uses_the_store_pool(self):
        rt, _sdk, native, board, _front, preload, store = self._window()
        rt.shutdown()
        try:
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_READING,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._park_wait["a"] = 1
            rt.request_finished("a", [3], 16)
            self.assertEqual(sum(1 for item in native.submits if item[0] == "d2h"), 1)
            self.assertFalse(any(not slot.freed and slot.purpose == "preload" and slot.req_id == "a" for slot in preload._slots))
            self.assertTrue(any(slot.purpose == "store" and slot.req_id == "a" and not slot.freed for slot in store._slots))
        finally:
            self._close(rt, preload, store, board)

    def test_leave_gpu_without_blocks_closes(self):
        rt, _sdk, native, board, _front, preload, store = self._window()
        rt.shutdown()
        try:
            with rt._mu:
                rt._reqs["c"] = _Req(req_id="c", generation=1, tokens=[1] * 16, phase=PHASE_PENDING)
            rt._slots["c"] = 1
            board.publish_at(1, 1, PHASE_PENDING, 0, 0)
            rt.preempt("c")
            with rt._mu:
                self.assertTrue(rt._reqs["c"].closed)
            self.assertEqual(board.get_at(1).phase, PHASE_CLOSED)
            self.assertFalse(any(item[0] == "d2h" for item in native.submits))
            self.assertFalse(any(not slot.freed and slot.req_id == "c" for slot in preload._slots))
        finally:
            self._close(rt, preload, store, board)

    def test_park_copy_finishes_before_return_unless_held(self):
        rt, _sdk, native, board, _front, preload, store = self._window()
        try:
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_HBM,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._slots["a"] = 0
            board.publish_at(0, 1, PHASE_HBM, 0, 0)
            native.poll_hold = True
            rt.preempt("a")
            self.assertEqual(sum(1 for item in native.submits if item[0] == "d2h"), 1)
            self.assertEqual(board.get_at(0).phase, PHASE_READING)
            native.poll_hold = False
            self.assertTrue(wait_until(lambda: board.get_at(0).phase == PHASE_PRELOADED))
        finally:
            self._close(rt, preload, store, board)

    def test_park_returns_preloaded_when_the_copy_is_not_held(self):
        rt, _sdk, _native, board, _front, preload, store = self._window()
        try:
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_HBM,
                    num_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                rt._reqs["a"] = req
            rt._slots["a"] = 0
            board.publish_at(0, 1, PHASE_HBM, 0, 0)
            rt.preempt("a")
            self.assertEqual(board.get_at(0).phase, PHASE_PRELOADED)
            self.assertEqual(rt.queue.get("a").status, STATUS_PRELOADED)
        finally:
            self._close(rt, preload, store, board)

    def test_h2d_then_park_allocates_new_preload_slots(self):
        rt, _sdk, native, board, _front, preload, store = self._window()
        rt.shutdown()
        try:
            old = preload.try_allocate(preload.slot_bytes, "preload", "a", 1)
            with rt._mu:
                req = _Req(
                    req_id="a",
                    generation=1,
                    tokens=[1] * 16,
                    phase=PHASE_H2D,
                    num_tokens=16,
                    hit_tokens=16,
                    block_ids=[3],
                )
                req.layer_base = {"0": 1000, "1": 2000}
                req._preload_slots = old
                rt._reqs["a"] = req
            rt._slots["a"] = 0
            board.publish_at(0, 1, PHASE_H2D, 0, 0)
            rt._jobs[11] = (CopyKind.H2D, "a", 1, old, {"sequence": 11})
            with native.lock:
                native.done.append({"sequence": 11, "status": 0, "direction": 1})
            rt._poll_copies()
            with rt._mu:
                self.assertIsNone(rt._reqs["a"]._preload_slots)
                self.assertEqual(rt._reqs["a"].phase, PHASE_HBM)
            self.assertTrue(old[0].freed)
            calls = {"n": 0}
            allocate = preload.try_allocate

            def wrapped(*args, **kwargs):
                calls["n"] += 1
                return allocate(*args, **kwargs)

            preload.try_allocate = wrapped
            rt.preempt("a")
            rt._poll_copies()
            parked = rt.queue.get("a")
            self.assertEqual(parked.status, STATUS_PRELOADED)
            with rt._mu:
                fresh = rt._reqs["a"]._preload_slots
            self.assertTrue(fresh)
            self.assertEqual(calls["n"], 1)
            self.assertFalse(fresh[0].freed)
            self.assertEqual(sum(1 for item in native.submits if item[0] == "d2h"), 1)
        finally:
            self._close(rt, preload, store, board)

    def test_one_submit_covers_131_blocks(self):
        rt, _sdk, native, _board, _front, preload, store = self._window()
        rt.shutdown()
        native.hold = True
        native.h2d_rcs = [1]
        generation = rt.queue.enqueue_lookup("p", {"tokens": [1] * 16}, 16)
        self.assertTrue(rt.queue.complete_lookup("p", generation, 16, 16))
        self.assertTrue(rt.queue.mark_reading("p", generation))
        self.assertTrue(rt.queue.mark_preloaded("p", generation, 16))
        payload = 128
        count = 131

        class _Slot:
            nbytes = payload * 2 * count
            buf = 0x1000
            freed = False

        with rt._mu:
            rt._reqs["p"] = _Req(
                req_id="p",
                generation=generation,
                tokens=[1] * (count * 16),
                phase=PHASE_PRELOADED,
                hit_tokens=count * 16,
                external_tokens=count * 16,
                block_ids=list(range(count)),
                layer_base={"0": 1000, "1": 5000},
            )
            rt._reqs["p"]._preload_slots = [_Slot()]
        rt._submit_preloaded_h2d("p", generation)
        self.assertEqual(len(self._h2d_submits(native)), 1)
        self.assertEqual(len(native.h2d_spans[-1]), 2)
        self.assertEqual(native.h2d_spans[-1][0]["bytes"], payload * count)
        self.assertEqual(rt.queue.get("p").status, STATUS_PRELOADED)
        self.assertNotIn("p", rt.transfer_results().failed_recving)
        native.h2d_rcs[:] = [0]
        rt._submit_preloaded_h2d("p", generation)
        self.assertEqual(len(self._h2d_submits(native)), 2)
        self.assertEqual(rt.queue.get("p").status, STATUS_H2D_IN_FLIGHT)
        self._close(rt, preload, store, _front.board)

    def test_clear_published_drops_failed_block_ids(self):
        rt, sdk, native, _board, front, preload, store = self._window()
        native.h2d_rcs = [2]
        sdk.scripted[1] = 32
        try:
            front.on_new_request("a", [1] * 32)
            rt.ingest(front.build_meta())
            self.assertTrue(wait_until(lambda: rt.matched_tokens("a") == (32, True)))
            rt.observe_layer("0", 1000, None)
            rt.observe_layer("1", 2000, None)
            self.assertTrue(rt.update_state_after_alloc("a", [3, 4], 32))
            self.assertTrue(wait_until(lambda: rt.queue.get("a").status == STATUS_FALLBACK_MISS))
            self.assertIn("a", rt.transfer_results().failed_recving)
            rt.clear_published()
            self.assertNotIn("a", rt.transfer_results().failed_recving)
        finally:
            self._close(rt, preload, store, front.board)


class _PinNative:
    def __init__(self, pin_rc=0, checks=None, error=None):
        self.pin_rc = pin_rc
        self.checks = 0 if checks is None else checks
        self.error = error
        self.frees = []
        self._n = 0

    def hostmem_alloc(self, nbytes):
        self._n += 1
        ptr = 0x300000 + self._n * 0x100000
        mapped = (int(nbytes) + 4095) & ~4095
        return ptr, mapped

    def hostmem_free(self, ptr, mapped):
        self.frees.append((int(ptr), int(mapped)))

    def host_pin(self, ptr, nbytes):
        del ptr, nbytes
        if self.error is not None:
            raise self.error
        return self.pin_rc

    def copy_async_check(self, ptr, nbytes, direction):
        del ptr, nbytes
        if isinstance(self.checks, dict):
            return int(self.checks[int(direction)])
        return int(self.checks)

    def host_unpin(self, ptr):
        del ptr
        return 0


def _import_ptr():
    import types

    def mod(name):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        return module

    if "vllm.distributed.kv_transfer.kv_connector.v1.base" not in sys.modules:
        mod("vllm")
        mod("vllm.distributed")
        mod("vllm.distributed.kv_transfer")
        mod("vllm.distributed.kv_transfer.kv_connector")
        mod("vllm.distributed.kv_transfer.kv_connector.v1")
        base = mod("vllm.distributed.kv_transfer.kv_connector.v1.base")

        class KVConnectorBase_V1:
            def __init__(self, *args, **kwargs):
                self._connector_metadata = None

            def bind_connector_metadata(self, connector_metadata) -> None:
                self._connector_metadata = connector_metadata

            def clear_connector_metadata(self) -> None:
                self._connector_metadata = None

            def has_connector_metadata(self) -> bool:
                return self._connector_metadata is not None

        class KVConnectorRole:
            SCHEDULER = "scheduler"

        base.KVConnectorBase_V1 = KVConnectorBase_V1
        base.KVConnectorRole = KVConnectorRole
        mod("vllm.v1")
        mod("vllm.v1.core")
        mod("vllm.v1.core.sched")
        output = mod("vllm.v1.core.sched.output")

        class SchedulerOutput:
            pass

        output.SchedulerOutput = SchedulerOutput
    from surdata_vllm.connector import _ptr

    return _ptr


class PinAndPtrTest(unittest.TestCase):
    def test_ptr_accepts_c_void_p_and_int(self):
        ptr = _import_ptr()
        self.assertEqual(ptr(ctypes.c_void_p(0x1234)), 0x1234)
        self.assertIsNone(ptr(ctypes.c_void_p(0)))
        self.assertIsNone(ptr(ctypes.c_void_p()))
        self.assertEqual(ptr(7), 7)
        self.assertEqual(ptr(0), 0)

    def test_one_page_lock_or_eperm(self):
        before = _vm_lck_kb()
        try:
            base, mapped = alloc_hostmem(4096)
        except OSError as exc:
            if exc.errno == errno.EPERM:
                return
            raise
        try:
            self.assertGreaterEqual(mapped, 4096)
            self.assertGreaterEqual(_vm_lck_kb(), before + mapped // 1024)
        finally:
            free_hostmem(base, mapped)

    def _runtime(self, native):
        return WorkerRuntime(
            sdk=FakeSdk(),
            native=native,
            policy="finish_drop",
            preload_bytes=4096,
            store_bytes=4096,
        )

    def _geometry(self):
        return KvGeometry(4, 16, 4, 2, 1)

    def test_pin_error_closes_pool_once(self):
        native = _PinNative(pin_rc=1)
        rt = self._runtime(native)
        with self.assertRaises(RuntimeError) as caught:
            rt.set_geometry(self._geometry())
        self.assertIn("rc=1", str(caught.exception))
        self.assertIn("bytes=", str(caught.exception))
        self.assertFalse(rt.preload_pool.host_pinned)
        self.assertEqual(len(native.frees), 1)
        rt.preload_pool.close()
        self.assertEqual(len(native.frees), 1)
        self.assertIsNone(rt.store_pool)

        native = _PinNative(error=OSError("register"))
        rt = self._runtime(native)
        with self.assertRaises(OSError):
            rt.set_geometry(self._geometry())
        self.assertFalse(rt.preload_pool.host_pinned)
        self.assertEqual(len(native.frees), 1)

    def test_sync_check_closes_that_pool(self):
        native = _PinNative(checks={1: 4, 2: 0})
        rt = self._runtime(native)
        with self.assertRaises(RuntimeError) as caught:
            rt.set_geometry(self._geometry())
        self.assertIn("rc=4", str(caught.exception))
        self.assertFalse(rt.preload_pool.host_pinned)
        self.assertEqual(len(native.frees), 1)
        self.assertIsNone(rt.store_pool)
        text = _pin_text()
        self.assertIn("check=h2d-sync rc=4", text)

        native = _PinNative(checks={1: 0, 2: 4})
        rt = self._runtime(native)
        with self.assertRaises(RuntimeError):
            rt.set_geometry(self._geometry())
        self.assertTrue(rt.preload_pool.host_pinned)
        self.assertFalse(rt.store_pool.host_pinned)
        self.assertEqual(len(native.frees), 1)
        text = _pin_text()
        self.assertIn("check=d2h-sync rc=4", text)
        self.assertIn("check=h2d-async", text)
        rt.preload_pool.close()
        rt.store_pool.close()
        self.assertEqual(len(native.frees), 2)

    def test_both_async_pins_and_close_frees_once(self):
        with open("/tmp/surdata_pin.txt", "w", encoding="utf-8") as handle:
            handle.write("rc=0 bytes=1\n")
        native = _PinNative(checks={1: 0, 2: 0})
        rt = self._runtime(native)
        rt.set_geometry(self._geometry())
        self.assertTrue(rt.preload_pool.host_pinned)
        self.assertTrue(rt.store_pool.host_pinned)
        text = _pin_text()
        self.assertNotIn("rc=0 bytes=1", text)
        self.assertIn("kind=preload alloc=anon-private mlock=1 pin_rc=0 check=h2d-async", text)
        self.assertIn("kind=store alloc=anon-private mlock=1 pin_rc=0 check=d2h-async", text)
        rt.preload_pool.close()
        rt.store_pool.close()
        self.assertEqual(len(native.frees), 2)
        rt.preload_pool.close()
        rt.store_pool.close()
        self.assertEqual(len(native.frees), 2)
        self.assertFalse(rt.preload_pool.host_pinned)
        self.assertFalse(rt.store_pool.host_pinned)


class _Dev:
    def __init__(self, kind, index):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return isinstance(other, _Dev) and self.type == other.type and self.index == other.index

    def __str__(self):
        if self.index is None:
            return self.type
        return "%s:%s" % (self.type, self.index)


class _Storage:
    def __init__(self, device):
        self.device = device


class _ViewTensor:
    def __init__(self, device):
        self.device = device
        self.storage = None

    def set_(self, storage, offset, shape, stride):
        del offset, shape, stride
        if self.device != storage.device:
            raise RuntimeError(
                'Attempted to set the storage of a tensor on device "%s" to a storage on different device "%s".'
                % (self.device, storage.device)
            )
        self.storage = storage


class _UnindexedCudaTorch:
    """torch 2.14: storage keeps ``cuda``, empty() lands on ``cuda:0``."""

    def __init__(self):
        self.uint8 = "u8"
        self.cuda = self
        self._C = self
        self.current_calls = 0
        self.storage_devices = []

    def current_device(self):
        self.current_calls += 1
        return 0

    def device(self, name, index=None):
        return _Dev(str(name), None if index is None else int(index))

    def _construct_storage_from_data_pointer(self, ptr, dev, nbytes):
        del ptr, nbytes
        self.storage_devices.append(dev)
        return _Storage(dev)

    def empty(self, count, dtype=None, device=None):
        del count, dtype
        dev = device
        if getattr(dev, "type", None) == "cuda" and getattr(dev, "index", None) is None:
            dev = _Dev("cuda", 0)
        return _ViewTensor(dev)


class ByteViewDeviceTest(unittest.TestCase):
    def test_cuda_view_uses_current_device_index(self):
        torch_mod = _UnindexedCudaTorch()
        view = _byte_view(torch_mod, 100, 16, "cuda")
        self.assertEqual(torch_mod.current_calls, 1)
        self.assertEqual(torch_mod.storage_devices[0].type, "cuda")
        self.assertEqual(torch_mod.storage_devices[0].index, 0)
        self.assertEqual(view.device, torch_mod.storage_devices[0])
        self.assertIs(view.storage.device, torch_mod.storage_devices[0])

    def test_cpu_view_does_not_read_current_device(self):
        torch_mod = _UnindexedCudaTorch()
        view = _byte_view(torch_mod, 200, 8, "cpu")
        self.assertEqual(torch_mod.current_calls, 0)
        self.assertEqual(view.device.type, "cpu")
        self.assertIsNone(view.device.index)
        self.assertEqual(torch_mod.storage_devices[0].type, "cpu")
        self.assertIsNone(torch_mod.storage_devices[0].index)


if __name__ == "__main__":
    unittest.main()
