"""ctypes calls into libsurdata_sdk. Unit tests use a fake SDK instead."""

import ctypes
import os
import threading
import time


class SdkError(RuntimeError):
    pass


def namespace_model_id(model: str) -> str:
    """Namespace identity is the model directory name, never a path."""
    name = os.path.basename(str(model or ""))
    return name or "m"


_COMPLETE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32)
_WAVE_COMPLETE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64)
_ERR_BUSY = -9
_ERR_STATE = -6
_ERR_INVALID = -1  # SURDATA_SDK_ERR_INVALID_PARAM
_KV_LAYOUT_MLA = 2
_WAVE_BLOCKS = 64
_WAVE_WAIT_S = 30.0


class InitOpts(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("kv_root", ctypes.c_char_p),
        ("host_only", ctypes.c_int),
        ("log_callback", ctypes.c_void_p),
        ("log_user", ctypes.c_void_p),
        ("rehash_load_factor_percent", ctypes.c_uint32),
        ("dirty_slot_ratio_percent", ctypes.c_uint32),
    ]


class HitInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("found", ctypes.c_int),
        ("best_idx", ctypes.c_uint32),
        ("matched_tokens", ctypes.c_uint32),
        ("matched_token_blocks", ctypes.c_uint32),
        ("unmatched_token_blocks", ctypes.c_uint32),
        ("logical_len", ctypes.c_uint64),
    ]


class PrefixFulfill(ctypes.Structure):
    """Same layout as surdata_sdk_prefix_fulfill_t. Count 0 with N == M is an empty gap."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("local_token_blocks", ctypes.c_uint32),
        ("fulfill_token_block_start", ctypes.c_uint32),
        ("fulfill_token_block_count", ctypes.c_uint32),
    ]


class LayerSlice(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("bytes", ctypes.c_size_t)]


class WriteArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("kv_block_index", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("layers", ctypes.POINTER(LayerSlice)),
        ("bytes_per_layer_slice", ctypes.c_size_t),
        ("sub_stripe_bytes", ctypes.c_uint64),
        ("preallocate_extent", ctypes.c_uint8),
        ("complete_fn", _COMPLETE),
        ("complete_user", ctypes.c_void_p),
        ("publish_on_complete", ctypes.c_uint8),
        ("publish_token_block_start", ctypes.c_uint32),
        ("publish_token_block_count", ctypes.c_uint32),
        ("kv_layout", ctypes.c_uint32),
        ("vllm_block_size", ctypes.c_uint32),
        ("elem_bytes", ctypes.c_uint64),
        ("tail_elems", ctypes.c_uint64),
        ("idempotency_key", ctypes.c_uint64),
        ("old_visible_len", ctypes.c_uint64),
        ("new_visible_len", ctypes.c_uint64),
        ("publish_reason", ctypes.c_uint32),
        ("file_ec_unit_bytes", ctypes.c_uint64),
    ]


class WaveArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("wave_kvb_count", ctypes.c_uint32),
        ("extent_bytes", ctypes.c_uint64),
        ("logical_offset", ctypes.c_uint64),
        ("sub_stripe_bytes", ctypes.c_uint64),
        ("complete_fn", _WAVE_COMPLETE),
        ("complete_user", ctypes.c_void_p),
        ("file_ec_unit_bytes", ctypes.c_uint64),
    ]


class Metric(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("align", ctypes.c_uint64),
        ("file_ec_unit", ctypes.c_uint64),
    ]


class DurableArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("token_block_start", ctypes.c_uint32),
        ("token_block_count", ctypes.c_uint32),
        ("kv_layout", ctypes.c_uint32),
        ("vllm_block_size", ctypes.c_uint32),
        ("elem_bytes", ctypes.c_uint64),
        ("tail_elems", ctypes.c_uint64),
        ("logical_lens", ctypes.POINTER(ctypes.c_uint64)),
        ("publish_reason", ctypes.c_uint32),
        ("physical_size_bytes", ctypes.c_uint64),
    ]


class _SaveSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.sdk_req = None
        self.hash_count = 0
        self.block_size = 0
        self.extent = 0
        self.durable = set()
        self.skipped = set()
        self.inflight = 0
        self.failed = False
        self.finished = False
        self.abandoned = False
        self.callback_depth = 0
        self.committed = False
        self.destroyed = False
        self.hit_blocks = 0
        self.token_count = 0
        self.parent_hit = None
        self.committing = False
        self.holds = []
        self.wave_ready = 0
        self.wave_inflight = False
        self.wave_event = None
        self.wave_error = 0
        self.wave_key = 0


class ReadArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("kv_block_index", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("layers", ctypes.POINTER(LayerSlice)),
        ("bytes_per_layer_slice", ctypes.c_size_t),
        ("sub_stripe_bytes", ctypes.c_uint64),
        ("complete_fn", _COMPLETE),
        ("complete_user", ctypes.c_void_p),
        ("file_ec_unit_bytes", ctypes.c_uint64),
    ]


def abi_sizes():
    return {
        "init": ctypes.sizeof(InitOpts),
        "hit": ctypes.sizeof(HitInfo),
        "slice": ctypes.sizeof(LayerSlice),
        "write": ctypes.sizeof(WriteArgs),
        "read": ctypes.sizeof(ReadArgs),
    }


class SdkBridge:
    def __init__(self, settings: dict):
        self.settings = settings
        self.geometry = None
        self._lib = None
        self._ns = ctypes.c_void_p()
        self._ns_lock = threading.Lock()
        self._bound = False
        self._hits = {}
        self._hashes = {}
        self._token_counts = {}
        self._tokens = {}
        self._publishing = set()
        self._cb_lock = threading.Lock()
        self._cbs = {}
        self._cb_next = 1
        self._c_complete = _COMPLETE(self._on_complete)
        self._c_wave = _WAVE_COMPLETE(self._on_wave)
        self._waves = {}
        self._wave_next = 1
        self._full_extent = 0
        self._file_ec = 0
        self._sessions = {}
        self._sessions_mu = threading.Lock()
        self._commit_wake = None

    def _load(self):
        if self._lib is not None:
            return self._lib
        path = os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so")
        self._lib = ctypes.CDLL(path)
        return self._lib

    def bind_owner(self):
        name = os.environ.get("SURDATA_LEASE_SHM", "")
        if not name:
            return
        lib = self._load()
        rc = lib.surdata_sdk_lease_bind(name.encode(), 1)
        if rc != 0:
            raise SdkError(f"lease_bind owner rc={rc}")

    def bind_client(self):
        name = os.environ.get("SURDATA_LEASE_SHM", "")
        if not name:
            return
        lib = self._load()
        rc = lib.surdata_sdk_lease_bind(name.encode(), 0)
        if rc != 0:
            raise SdkError(f"lease_bind client rc={rc}")

    def preload_enqueue(self, request_id, epoch, token_ids, block_tokens, bytes_per_token,
                        layout_signature, shard_id):
        raise SdkError("SdkBridge.preload_enqueue needs a live namespace; tests use FakeSdk")

    def preload_snapshot(self, request_id, epoch):
        raise SdkError("SdkBridge.preload_snapshot needs a live namespace; tests use FakeSdk")

    def try_hold_read(self, request_id, epoch, generation):
        raise SdkError("SdkBridge.try_hold_read needs a live namespace; tests use FakeSdk")

    def note_tokens(self, req_id, tokens) -> None:
        key = str(req_id)
        self._tokens[key] = [int(tok) for tok in tokens]
        self._publishing.add(key)

    def note_hit_blocks(self, req_id, generation, blocks) -> None:
        """Blocks already stored by an earlier request. This request writes after them."""
        sess = self._session(req_id, generation)
        with sess.lock:
            sess.hit_blocks = max(int(sess.hit_blocks), int(blocks))

    def prefix_may_arrive(self, req_id, tokens) -> bool:
        """True while another live request can still publish a shared prefix."""
        tokens = [int(tok) for tok in tokens]
        mine = str(req_id)
        for other in list(self._publishing):
            if other == mine:
                continue
            prompt = self._tokens.get(other) or []
            if not prompt or not tokens:
                continue
            n = min(len(prompt), len(tokens))
            if prompt[:n] == tokens[:n]:
                return True
        return False

    def _finish_publish(self, req_id) -> None:
        self._publishing.discard(str(req_id))

    def lookup_hit(self, req_id, tokens, block_tokens, generation) -> int:
        """Confirmed miss is 0. SDK errors stay pending for the caller to retry."""
        self.note_tokens(req_id, tokens)
        try:
            return self._lookup_hit(str(req_id), list(tokens), int(block_tokens), int(generation))
        except (SdkError, OSError, AttributeError) as exc:
            raise SdkError(str(exc)) from exc

    def drop_lookup(self, req_id, generation) -> None:
        hit = self._hits.pop((str(req_id), int(generation)), None)
        if hit:
            self._destroy_hit(hit)

    def submit_read(self, req_id, generation, hit_tokens, slots, on_done) -> int:
        """Submit surdata_sdk_read_kv_block_async and return. The callback runs later."""
        try:
            return self._submit_read(str(req_id), int(generation), int(hit_tokens), slots, on_done)
        except SdkError as exc:
            if "busy" in str(exc).lower():
                return _ERR_BUSY
            return -1
        except (OSError, AttributeError):
            return -1

    def read_kvb(self, req_id, generation, hit_tokens, slots) -> bool:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                return self._read_slots(str(req_id), int(generation), int(hit_tokens), slots)
            except SdkError as exc:
                if "busy" not in str(exc).lower() or time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
            except (OSError, AttributeError):
                return False

    def enqueue_save(
        self,
        req_id,
        generation,
        valid_tokens,
        block_tokens,
        slots,
        token_index,
        on_done,
        block_index=0,
        block_count=1,
    ) -> int:
        try:
            return self._enqueue_save(
                str(req_id),
                int(generation),
                int(valid_tokens),
                int(block_tokens),
                slots,
                int(token_index),
                on_done,
                int(block_index),
                int(block_count),
            )
        except (SdkError, OSError, AttributeError):
            return _ERR_BUSY

    def _on_complete(self, user, status, index):
        key = self._user_key(user)
        with self._cb_lock:
            fn = self._cbs.pop(key, None)
        if fn is not None:
            fn(int(status), int(index))

    def _on_wave(self, user, status, nbytes):
        del nbytes
        key = self._user_key(user)
        with self._cb_lock:
            box = self._waves.pop(key, None)
        if box is None:
            return
        sess = box["sess"]
        with sess.lock:
            if int(status) == 0:
                sess.wave_ready = int(box["end"])
                sess.wave_error = 0
            else:
                sess.wave_error = int(status)
            sess.wave_inflight = False
            ready = self._session_idle(sess)
        box["event"].set()
        if ready:
            self._wake_commit()

    def _full_block_extent(self, lib) -> int:
        if self._full_extent > 0:
            return self._full_extent
        geo = self.geometry
        if geo is None:
            raise SdkError("save before geometry")
        metric = Metric()
        metric.struct_size = ctypes.sizeof(Metric)
        rc = int(lib.surdata_sdk_get_metric(ctypes.byref(metric)))
        if rc != 0:
            raise SdkError("get_metric rc=%s" % rc)
        fec = int(metric.file_ec_unit) or int(metric.align)
        if fec <= 0:
            raise SdkError("file_ec unavailable")
        payload = int(geo.num_layers) * int(geo.layer_payload(geo.block_size))
        extent = int(lib.surdata_sdk_kv_block_extent_bytes(int(payload), int(fec)))
        if extent <= 0 or extent % fec != 0:
            raise SdkError("extent %s not aligned to file_ec %s" % (extent, fec))
        self._full_extent = extent
        self._file_ec = fec
        return extent

    def _issue_wave(self, lib, req, sess, start: int, extent: int, window: int) -> int:
        end = int(start) + int(window)
        event = threading.Event()
        with self._cb_lock:
            key = self._wave_next
            self._wave_next += 1
            self._waves[key] = {"sess": sess, "end": end, "event": event}
        with sess.lock:
            sess.wave_inflight = True
            sess.wave_event = event
            sess.wave_error = 0
            sess.wave_key = key
        args = WaveArgs()
        args.struct_size = ctypes.sizeof(WaveArgs)
        args.version = 1
        args.wave_kvb_count = _WAVE_BLOCKS
        args.extent_bytes = int(extent)
        args.logical_offset = int(start)
        args.sub_stripe_bytes = 0
        args.complete_fn = self._c_wave
        args.complete_user = ctypes.c_void_p(key)
        args.file_ec_unit_bytes = int(self._file_ec)
        rc = int(lib.surdata_sdk_prepare_external_kv_wave_async(req, ctypes.byref(args)))
        if rc != 0:
            with self._cb_lock:
                self._waves.pop(key, None)
            with sess.lock:
                sess.wave_inflight = False
                sess.wave_error = rc
            event.set()
        return rc

    def _ensure_wave(self, lib, req, sess, block_index: int) -> int:
        extent = self._full_block_extent(lib)
        window = _WAVE_BLOCKS * extent
        block_off = int(block_index) * extent
        block_end = block_off + extent
        while True:
            event = None
            start = None
            with sess.lock:
                if sess.wave_error:
                    return int(sess.wave_error)
                covered = block_end <= sess.wave_ready
                if covered and not sess.wave_inflight and sess.wave_ready > 0 and block_off >= sess.wave_ready - window // 2:
                    start = int(sess.wave_ready)
                elif not covered and not sess.wave_inflight:
                    start = int(sess.wave_ready)
                elif not covered:
                    event = sess.wave_event
            if start is not None:
                rc = self._issue_wave(lib, req, sess, start, extent, window)
                if rc != 0:
                    return rc
                if covered:
                    return 0
                continue
            if event is not None:
                if not event.wait(_WAVE_WAIT_S):
                    with sess.lock:
                        if sess.wave_error == 0 and sess.wave_inflight:
                            sess.wave_error = _ERR_INVALID
                        if sess.wave_inflight:
                            return _ERR_INVALID
                    continue
                continue
            return 0

    def _park(self, fn):
        with self._cb_lock:
            key = self._cb_next
            self._cb_next += 1
            self._cbs[key] = fn
            return key

    def _lookup_hit(self, req_id, tokens, block_tokens, generation) -> int:
        count = len(tokens) // max(1, block_tokens)
        if count <= 0:
            return 0
        lib = self._load()
        ns = self._namespace()
        self._bind_fns(lib)
        packed = self._hash_tokens(lib, tokens, block_tokens)
        if packed is None:
            return 0
        self._hashes[req_id] = packed
        hashes, out_count, _tokens = packed
        hit = ctypes.c_void_p()
        rc = lib.surdata_sdk_lookup_prefix_hit(ns, ctypes.addressof(hashes), int(out_count), ctypes.byref(hit))
        if rc != 0 or not hit.value:
            raise SdkError("lookup rc=%s" % rc)
        info = HitInfo()
        info.struct_size = ctypes.sizeof(HitInfo)
        info.version = 1
        rc = lib.surdata_sdk_prefix_hit_get_info(hit, ctypes.byref(info))
        if rc != 0:
            lib.surdata_sdk_prefix_hit_destroy(hit)
            raise SdkError("hit info rc=%s" % rc)
        if not info.found:
            lib.surdata_sdk_prefix_hit_destroy(hit)
            return 0
        previous = self._hits.pop((req_id, generation), None)
        if previous:
            self._destroy_hit(previous)
        self._hits[(req_id, generation)] = hit
        # matched_tokens is in the namespace unit (LMCache chunk is 256).
        # matched_token_blocks is the hash count. This connector's hashes are
        # one per vLLM block, so the aligned count is blocks times that size.
        # logical_len is the stored token length. A short tail sits inside the
        # last block; a namespace-sized matched_tokens is not that length.
        block_tokens = max(1, int(block_tokens))
        blocks = int(info.matched_token_blocks)
        if blocks <= 0 and int(info.matched_tokens) > 0:
            blocks = int(info.matched_tokens) // block_tokens
        aligned = blocks * block_tokens
        logical = int(getattr(info, "logical_len", 0) or 0)
        if blocks > 0 and logical > 0 and aligned - block_tokens < logical <= aligned:
            return int(logical)
        if blocks <= 0 and 0 < logical < block_tokens:
            return int(logical)
        return aligned

    def _namespace(self):
        if self._ns.value:
            return self._ns
        with self._ns_lock:
            if self._ns.value:
                return self._ns
            lib = self._load()
            self._bind_fns(lib)
            opts = InitOpts()
            opts.struct_size = ctypes.sizeof(InitOpts)
            opts.version = 1
            root = os.environ.get("SURDATA_KV_ROOT", "")
            opts.kv_root = root.encode() if root else None
            opts.host_only = 1
            rc = int(lib.surdata_sdk_init(ctypes.byref(opts)))
            if rc not in (0, -3, -6):
                raise SdkError("sdk init rc=%s" % rc)
            rc = int(lib.surdata_sdk_connect_storage_from_env())
            if rc != 0:
                raise SdkError("connect_storage rc=%s" % rc)
            model = str(self.settings.get("model", "m")).encode()
            version = str(self.settings.get("model_version", "v1")).encode()
            layout = str(self.settings.get("layout_signature", "mla")).encode()
            rc = int(lib.surdata_sdk_namespace_open(model, version, layout, ctypes.byref(self._ns)))
            if rc != 0 or not self._ns.value:
                raise SdkError("namespace_open rc=%s" % rc)
            return self._ns

    def _bind_fns(self, lib):
        if self._bound:
            return
        lib.surdata_sdk_init.argtypes = [ctypes.POINTER(InitOpts)]
        lib.surdata_sdk_init.restype = ctypes.c_int
        lib.surdata_sdk_connect_storage_from_env.argtypes = []
        lib.surdata_sdk_connect_storage_from_env.restype = ctypes.c_int
        lib.surdata_sdk_namespace_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.surdata_sdk_namespace_open.restype = ctypes.c_int
        lib.surdata_sdk_request_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.surdata_sdk_request_create.restype = ctypes.c_int
        lib.surdata_sdk_request_destroy.argtypes = [ctypes.c_void_p]
        lib.surdata_sdk_request_destroy.restype = ctypes.c_int
        lib.surdata_sdk_prefix_hashes_from_tokens.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.surdata_sdk_prefix_hashes_from_tokens.restype = ctypes.c_int
        lib.surdata_sdk_lookup_prefix_hit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.surdata_sdk_lookup_prefix_hit.restype = ctypes.c_int
        lib.surdata_sdk_prefix_hit_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(HitInfo)]
        lib.surdata_sdk_prefix_hit_get_info.restype = ctypes.c_int
        lib.surdata_sdk_prefix_hit_destroy.argtypes = [ctypes.c_void_p]
        lib.surdata_sdk_request_bind_prefix_hit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(PrefixFulfill),
        ]
        lib.surdata_sdk_request_bind_prefix_hit.restype = ctypes.c_int
        lib.surdata_sdk_request_set_prefix_hashes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.surdata_sdk_request_set_prefix_hashes.restype = ctypes.c_int
        lib.surdata_sdk_read_kv_block_async.argtypes = [ctypes.c_void_p, ctypes.POINTER(ReadArgs)]
        lib.surdata_sdk_read_kv_block_async.restype = ctypes.c_int
        lib.surdata_sdk_submit_external_kv_block_async.argtypes = [ctypes.c_void_p, ctypes.POINTER(WriteArgs)]
        lib.surdata_sdk_submit_external_kv_block_async.restype = ctypes.c_int
        lib.surdata_sdk_commit_kv_durable_prefix.argtypes = [ctypes.c_void_p, ctypes.POINTER(DurableArgs)]
        lib.surdata_sdk_commit_kv_durable_prefix.restype = ctypes.c_int
        lib.surdata_sdk_request_kv_block_extent.argtypes = [ctypes.c_void_p]
        lib.surdata_sdk_request_kv_block_extent.restype = ctypes.c_uint64
        lib.surdata_sdk_get_metric.argtypes = [ctypes.POINTER(Metric)]
        lib.surdata_sdk_get_metric.restype = ctypes.c_int
        lib.surdata_sdk_kv_block_extent_bytes.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.surdata_sdk_kv_block_extent_bytes.restype = ctypes.c_uint64
        lib.surdata_sdk_prepare_external_kv_wave_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(WaveArgs),
        ]
        lib.surdata_sdk_prepare_external_kv_wave_async.restype = ctypes.c_int
        self._bound = True

    def set_commit_wake(self, wake) -> None:
        self._commit_wake = wake

    def session_locked(self, req_id, generation) -> bool:
        with self._sessions_mu:
            sess = self._sessions.get((str(req_id), int(generation)))
        if sess is None:
            return False
        return sess.lock.locked()

    def _wake_commit(self) -> None:
        wake = self._commit_wake
        if wake is not None:
            wake()

    def _session(self, req_id, generation) -> _SaveSession:
        key = (str(req_id), int(generation))
        with self._sessions_mu:
            sess = self._sessions.get(key)
            if sess is None:
                sess = _SaveSession()
                self._sessions[key] = sess
            return sess

    def _session_idle(self, sess: _SaveSession) -> bool:
        if sess.destroyed or sess.committing:
            return False
        if sess.inflight or sess.callback_depth or sess.wave_inflight:
            return False
        return bool(sess.abandoned or sess.finished)

    def _commit_action(self, sess: _SaveSession):
        if sess.destroyed or sess.committing:
            return None, 0
        if sess.inflight or sess.callback_depth or sess.wave_inflight:
            return None, 0
        if sess.abandoned:
            return "abandon", 0
        if not sess.finished:
            return None, 0
        if sess.failed:
            return "destroy", 0
        start = int(sess.hit_blocks)
        limit = int(sess.hash_count)
        count = 0
        idx = start
        while idx < limit and idx not in sess.skipped and idx in sess.durable:
            count += 1
            idx += 1
        if count == 0:
            return "destroy", 0
        return "commit", count

    def apply_notes(self, notes) -> bool:
        """Store thread only. Notes are skip, abandon, or finished."""
        ready = False
        for req_id, generation, op, token_index in notes:
            req_id = str(req_id)
            generation = int(generation)
            if op == "abandon":
                self._hashes.pop(req_id, None)
                self._finish_publish(req_id)
                self.drop_lookup(req_id, generation)
            sess = self._session(req_id, generation)
            with sess.lock:
                if op == "skip":
                    sess.skipped.add(int(token_index))
                elif op == "abandon":
                    sess.abandoned = True
                elif op == "finished":
                    sess.finished = True
                if self._session_idle(sess):
                    ready = True
        return ready

    def commit_ready(self, settled=None) -> None:
        with self._sessions_mu:
            items = list(self._sessions.items())
        for key, sess in items:
            req_id, generation = key
            with sess.lock:
                action, count = self._commit_action(sess)
            if action is None:
                continue
            if action != "abandon" and settled is not None and not settled(req_id, generation):
                continue
            with sess.lock:
                action, count = self._commit_action(sess)
                if action is None:
                    continue
                sess.committing = True
                req = sess.sdk_req
                block_size = int(sess.block_size)
                extent = int(sess.extent)
            if action != "abandon" and settled is not None and not settled(req_id, generation):
                with sess.lock:
                    sess.committing = False
                continue
            with sess.lock:
                if sess.inflight or sess.callback_depth or sess.wave_inflight:
                    sess.committing = False
                    continue
                req = sess.sdk_req
                block_size = int(sess.block_size)
                extent = int(sess.extent)
            self._close_session(sess, req, action, count, block_size, extent, req_id)

    def _close_session(self, sess, req, action, count, block_size, extent, req_id) -> None:
        rc = 0
        try:
            if action == "commit" and req is not None and getattr(req, "value", None):
                if extent == 0:
                    extent = int(self._load().surdata_sdk_request_kv_block_extent(req))
                if extent == 0:
                    with sess.lock:
                        sess.committing = False
                    return
                start = int(sess.hit_blocks)
                token_count = int(sess.token_count)
                rc = self._commit_prefix(req, count, block_size, extent, start, token_count)
            terminal = (
                action != "commit"
                or rc == 0
                or rc == _ERR_STATE
                or rc == _ERR_INVALID
            )
            if terminal:
                self._destroy_req(req)
                with sess.lock:
                    parent_hit = sess.parent_hit
                    sess.parent_hit = None
                    sess.destroyed = True
                    sess.committed = action == "commit" and rc == 0
                    if action == "commit" and rc != 0:
                        sess.failed = True
                    sess.sdk_req = None
                    sess.holds.clear()
                self._destroy_hit(parent_hit)
                self._finish_publish(req_id)
            else:
                with sess.lock:
                    sess.committing = False
        except Exception:
            with sess.lock:
                sess.committing = False
            raise

    def _commit_prefix(self, req, count, block_size, extent, start=0, token_count=0) -> int:
        lib = self._load()
        self._bind_fns(lib)
        geo = self.geometry
        args = DurableArgs()
        args.struct_size = ctypes.sizeof(DurableArgs)
        args.version = 2
        args.token_block_start = int(start)
        args.token_block_count = int(count)
        args.kv_layout = _KV_LAYOUT_MLA
        if block_size <= 0 and geo is not None:
            block_size = int(geo.block_size)
        args.vllm_block_size = int(block_size)
        args.elem_bytes = 0 if geo is None else int(geo.itemsize)
        args.tail_elems = 0 if geo is None else int(geo.latent)
        args.publish_reason = 3
        args.physical_size_bytes = int(count) * int(extent)
        base = int(start)
        block = int(args.vllm_block_size)
        values = []
        for index in range(int(count)):
            logical = (base + index + 1) * block
            if int(token_count) > 0 and logical > int(token_count):
                logical = int(token_count)
            values.append(logical)
        lens = (ctypes.c_uint64 * int(count))(*values)
        args.logical_lens = ctypes.cast(lens, ctypes.POINTER(ctypes.c_uint64))
        return int(lib.surdata_sdk_commit_kv_durable_prefix(req, ctypes.byref(args)))

    def _request(self):
        lib = self._load()
        req = ctypes.c_void_p()
        rc = int(lib.surdata_sdk_request_create(self._namespace(), 0, ctypes.byref(req)))
        if rc == _ERR_BUSY:
            raise SdkError("request_create busy")
        if rc != 0 or not req.value:
            raise SdkError("request_create rc=%s" % rc)
        return req

    def _destroy_hit(self, hit):
        lib = self._load()
        if lib is not None and hit is not None and getattr(hit, "value", None):
            lib.surdata_sdk_prefix_hit_destroy(hit)

    def _destroy_req(self, req):
        lib = self._load()
        if lib is not None and req is not None and getattr(req, "value", None):
            lib.surdata_sdk_request_destroy(req)

    def _slices(self, slots, base, payload, layers):
        """Layer slices inside one compact KVB. The stride is ``payload``."""
        from surdata_vllm.streams import slot_addr

        buf = (LayerSlice * layers)()
        step = int(payload)
        for layer in range(layers):
            buf[layer].data = slot_addr(slots, int(base) + layer * step)
            buf[layer].bytes = step
        return buf

    def _slices_at(self, slots, block_index, block_count, payload, layers, nbytes=None):
        """One block's layers inside a layer-major run of ``block_count`` blocks.

        ``payload`` is the full-block stride. ``nbytes`` is the bytes actually
        read or written for this block. A short tail uses a smaller ``nbytes``
        at the same stride so the next block stays aligned.
        """
        from surdata_vllm.streams import slot_addr

        buf = (LayerSlice * layers)()
        step = int(payload)
        width = step if nbytes is None else int(nbytes)
        count = max(1, int(block_count))
        index = int(block_index)
        for layer in range(layers):
            buf[layer].data = slot_addr(slots, (layer * count + index) * step)
            buf[layer].bytes = width
        return buf

    def _hash_tokens(self, lib, tokens, block_tokens):
        block = max(1, int(block_tokens))
        count = (len(tokens) + block - 1) // block
        if count <= 0:
            return None
        hashes = (ctypes.c_uint8 * (16 * count))()
        out_count = ctypes.c_uint32()
        token_buf = (ctypes.c_uint32 * len(tokens))(*[int(tok) & 0xFFFFFFFF for tok in tokens])
        rc = lib.surdata_sdk_prefix_hashes_from_tokens(
            token_buf, len(tokens), int(block_tokens), ctypes.addressof(hashes), count, ctypes.byref(out_count)
        )
        if rc != 0:
            raise SdkError("prefix hash rc=%s" % rc)
        return hashes, int(out_count.value), list(tokens)

    def _bind_hashes(self, lib, req, req_id):
        packed = self._hashes.get(req_id)
        if packed is None:
            tokens = self._tokens.get(req_id) or []
            block = 1 if self.geometry is None else int(self.geometry.block_size)
            packed = self._hash_tokens(lib, tokens, block)
            if packed is None:
                return
            self._hashes[req_id] = packed
        buf, count, bound_tokens = packed
        rc = int(lib.surdata_sdk_request_set_prefix_hashes(req, ctypes.addressof(buf), int(count)))
        if rc == 0:
            self._token_counts[str(req_id)] = len(bound_tokens)
        if rc != 0:
            raise SdkError("set_prefix_hashes rc=%s" % rc)

    def _bind_parent_prefix(self, lib, req, req_id, hit_blocks, block_tokens):
        """Point this new file at the file that already holds the matched prefix.

        The lookup hit is not the preload hit. It stays alive until the save
        request is destroyed, which is after ensure_append_file has copied
        prev_file_id and prev_len. A miss or a length mismatch raises before
        any wave, so the new file is not created with prev_file_id 0.
        """
        tokens = self._tokens.get(str(req_id)) or []
        block_tokens = max(1, int(block_tokens))
        packed = self._hash_tokens(lib, tokens, block_tokens)
        if packed is None:
            raise SdkError("parent prefix has no hashes")
        hashes, out_count, _tokens = packed
        hit = ctypes.c_void_p()
        rc = int(
            lib.surdata_sdk_lookup_prefix_hit(
                self._namespace(), ctypes.addressof(hashes), int(out_count), ctypes.byref(hit)
            )
        )
        if rc != 0 or not getattr(hit, "value", None):
            raise SdkError("parent lookup rc=%s" % rc)
        info = HitInfo()
        info.struct_size = ctypes.sizeof(HitInfo)
        info.version = 1
        rc = int(lib.surdata_sdk_prefix_hit_get_info(hit, ctypes.byref(info)))
        if rc != 0:
            self._destroy_hit(hit)
            raise SdkError("parent hit info rc=%s" % rc)
        matched = int(info.matched_token_blocks)
        if matched == 0 and int(info.matched_tokens) > 0:
            matched = int(info.matched_tokens) // block_tokens
        if not int(info.found) or matched != int(hit_blocks):
            self._destroy_hit(hit)
            raise SdkError("parent hit blocks %s != %s" % (matched, hit_blocks))
        ful = PrefixFulfill()
        ful.struct_size = ctypes.sizeof(PrefixFulfill)
        ful.version = 1
        ful.local_token_blocks = int(hit_blocks)
        ful.fulfill_token_block_start = int(hit_blocks)
        ful.fulfill_token_block_count = 0
        rc = int(lib.surdata_sdk_request_bind_prefix_hit(req, hit, ctypes.byref(ful)))
        if rc != 0:
            self._destroy_hit(hit)
            raise SdkError("bind parent rc=%s" % rc)
        return hit

    def _submit_read(self, req_id, generation, hit_tokens, slots, on_done) -> int:
        geo = self.geometry
        hit = self._hits.pop((req_id, generation), None)
        if geo is None or hit is None:
            if hit is not None:
                self._destroy_hit(hit)
            raise SdkError("preload has no hit")
        lib = self._load()
        req = self._request()
        if int(lib.surdata_sdk_request_bind_prefix_hit(req, hit, None)) != 0:
            self._destroy_hit(hit)
            self._destroy_req(req)
            return -1
        block_size = int(geo.block_size)
        blocks = max(1, (int(hit_tokens) + block_size - 1) // block_size)
        payload = geo.layer_payload(block_size)
        state = {"left": 0, "failed": 0, "issuing": True, "done": False, "holds": []}
        lock = threading.Lock()

        def finish():
            with lock:
                if state["done"]:
                    return
                state["done"] = True
                status = int(state["failed"])
            try:
                on_done(status)
            finally:
                state["holds"].clear()
                self._destroy_hit(hit)
                self._destroy_req(req)

        def one(status, index):
            del index
            with lock:
                if int(status) != 0 and state["failed"] == 0:
                    state["failed"] = int(status)
                state["left"] -= 1
                ready = (not state["issuing"]) and state["left"] <= 0
            if ready:
                finish()

        submitted = 0
        for block in range(blocks):
            valid = block_size
            if block + 1 == blocks:
                rem = int(hit_tokens) % block_size
                if rem:
                    valid = rem
            nbytes = geo.layer_payload(valid)
            slices = self._slices_at(slots, block, blocks, payload, geo.num_layers, nbytes)
            with lock:
                state["left"] += 1
            try:
                rc, held = self._submit(
                    lib, req, block, slices, nbytes, False, 0, False, valid, one
                )
            except SdkError:
                with lock:
                    state["left"] -= 1
                rc = _ERR_BUSY
                held = None
            if rc != 0:
                with lock:
                    state["left"] -= 1
                    if state["failed"] == 0:
                        state["failed"] = int(rc)
                if submitted == 0:
                    self._destroy_hit(hit)
                    self._destroy_req(req)
                    return int(rc)
                break
            state["holds"].append(held)
            submitted += 1
        with lock:
            state["issuing"] = False
            ready = state["left"] <= 0
        if submitted == 0:
            return -1
        if ready:
            finish()
        return 0

    def _read_slots(self, req_id, generation, hit_tokens, slots) -> bool:
        geo = self.geometry
        hit = self._hits.pop((req_id, generation), None)
        if geo is None or hit is None:
            if hit is not None:
                self._destroy_hit(hit)
            raise SdkError("preload has no hit")
        lib = self._load()
        req = self._request()
        try:
            if int(lib.surdata_sdk_request_bind_prefix_hit(req, hit, None)) != 0:
                return False
            block_size = int(geo.block_size)
            blocks = max(1, (int(hit_tokens) + block_size - 1) // block_size)
            stride = geo.layer_payload(block_size)
            for block in range(blocks):
                valid = block_size
                if block + 1 == blocks:
                    rem = int(hit_tokens) % block_size
                    if rem:
                        valid = rem
                nbytes = geo.layer_payload(valid)
                slices = self._slices_at(slots, block, blocks, stride, geo.num_layers, nbytes)
                if not self._wait_io(lib, req, block, slices, nbytes, False, 0, valid):
                    return False
            return True
        finally:
            self._destroy_hit(hit)
            self._destroy_req(req)

    def _enqueue_save(
        self, req_id, generation, valid_tokens, block_tokens, slots, token_index, on_done, block_index=0, block_count=1
    ) -> int:
        geo = self.geometry
        if geo is None:
            raise SdkError("save before geometry")
        lib = self._load()
        self._namespace()
        self._bind_fns(lib)
        req_id = str(req_id)
        generation = int(generation)
        token_index = int(token_index)
        sess = self._session(req_id, generation)
        skip_hit = False
        hit_blocks = 0
        with sess.lock:
            if sess.abandoned or sess.destroyed or sess.committing:
                return _ERR_INVALID
            hit_blocks = int(sess.hit_blocks)
            if token_index < hit_blocks:
                skip_hit = True
            else:
                local_index = token_index - hit_blocks
                need_req = sess.sdk_req is None
        if skip_hit:
            if on_done is not None:
                on_done(0)
            return 0
        if need_req:
            req = self._request()
            parent_hit = None
            try:
                self._bind_hashes(lib, req, req_id)
                packed = self._hashes.get(req_id)
                hash_count = int(packed[1]) if packed else 0
                if hit_blocks > 0:
                    parent_hit = self._bind_parent_prefix(lib, req, req_id, hit_blocks, block_tokens)
            except SdkError:
                self._destroy_hit(parent_hit)
                self._destroy_req(req)
                if hit_blocks > 0:
                    return _ERR_INVALID
                raise
            except Exception:
                self._destroy_hit(parent_hit)
                self._destroy_req(req)
                raise
            with sess.lock:
                if sess.abandoned or sess.destroyed or sess.committing:
                    self._destroy_hit(parent_hit)
                    self._destroy_req(req)
                    return _ERR_INVALID
                if sess.sdk_req is None:
                    sess.sdk_req = req
                    sess.parent_hit = parent_hit
                    sess.hash_count = hash_count
                    sess.token_count = int(self._token_counts.get(req_id, 0))
                    sess.block_size = int(block_tokens)
                else:
                    self._destroy_hit(parent_hit)
                    self._destroy_req(req)
        with sess.lock:
            if sess.abandoned or sess.destroyed or sess.committing or sess.sdk_req is None:
                return _ERR_INVALID
            sess.inflight += 1
            req = sess.sdk_req
        payload = geo.layer_payload(valid_tokens if valid_tokens > 0 else block_tokens)
        count = max(1, int(block_count))
        if int(valid_tokens) >= int(geo.block_size) and count > 1:
            slices = self._slices_at(slots, int(block_index), count, geo.layer_payload(geo.block_size), geo.num_layers)
        else:
            slices = self._slices(slots, 0, payload, geo.num_layers)
        box = {"hold": None, "done": False}

        def done(status, index):
            del index
            with sess.lock:
                sess.callback_depth += 1
                if int(status) == 0:
                    sess.durable.add(token_index)
                else:
                    sess.failed = True
                if sess.inflight > 0:
                    sess.inflight -= 1
            try:
                on_done(int(status))
            finally:
                with sess.lock:
                    hold = box["hold"]
                    if hold is None:
                        box["done"] = True
                    elif hold in sess.holds:
                        sess.holds.remove(hold)
                    sess.callback_depth -= 1
                    ready = self._session_idle(sess)
            if ready:
                self._wake_commit()

        try:
            wave_rc = self._ensure_wave(lib, req, sess, local_index)
        except SdkError:
            with sess.lock:
                if sess.inflight > 0:
                    sess.inflight -= 1
            raise
        if wave_rc != 0:
            with sess.lock:
                if sess.inflight > 0:
                    sess.inflight -= 1
            return wave_rc
        try:
            rc, held = self._submit(
                lib, req, local_index, slices, payload, True, 0, False, valid_tokens, done
            )
        except SdkError:
            with sess.lock:
                if sess.inflight > 0:
                    sess.inflight -= 1
            raise
        if rc != 0:
            with sess.lock:
                if sess.inflight > 0:
                    sess.inflight -= 1
            return rc
        box["hold"] = held
        with sess.lock:
            if not box["done"]:
                sess.holds.append(held)
            if sess.extent == 0:
                sess.extent = int(lib.surdata_sdk_request_kv_block_extent(req))
        return 0

    def _wait_io(self, lib, req, block, slices, payload, write, token_index, valid_tokens) -> bool:
        event = threading.Event()
        box = {"status": -1}

        def done(status, index):
            del index
            box["status"] = status
            event.set()

        rc, held = self._submit(lib, req, block, slices, payload, write, token_index, False, valid_tokens, done)
        box["hold"] = held
        if rc == _ERR_BUSY:
            raise SdkError("kv block busy")
        if rc != 0:
            return False
        if not event.wait(30):
            return False
        box["hold"] = None
        return box["status"] == 0

    def _submit(self, lib, req, block, slices, payload, write, token_index, publish, valid_tokens, done):
        key = self._park(done)
        held = {"slices": slices}
        if write:
            args = WriteArgs()
            args.struct_size = ctypes.sizeof(WriteArgs)
            args.version = 1
            args.kv_block_index = int(block)
            args.layer_count = len(slices)
            args.layers = slices
            args.bytes_per_layer_slice = int(payload)
            args.sub_stripe_bytes = 0
            args.preallocate_extent = 0
            args.complete_fn = self._c_complete
            args.complete_user = ctypes.c_void_p(key)
            args.publish_on_complete = 1 if publish else 0
            args.publish_token_block_start = int(token_index) if publish else 0
            args.publish_token_block_count = 1 if publish else 0
            args.kv_layout = _KV_LAYOUT_MLA
            args.vllm_block_size = 0 if self.geometry is None else int(self.geometry.block_size)
            args.elem_bytes = 0 if self.geometry is None else int(self.geometry.itemsize)
            args.tail_elems = 0 if self.geometry is None else int(self.geometry.latent)
            args.idempotency_key = int(token_index) & 0xFFFFFFFFFFFFFFFF
            args.old_visible_len = (int(token_index) * args.vllm_block_size) if publish else 0
            if publish:
                args.new_visible_len = args.old_visible_len + args.vllm_block_size
            else:
                args.new_visible_len = int(valid_tokens)
            args.publish_reason = 3
            args.file_ec_unit_bytes = 0
            held["args"] = args
            rc = int(lib.surdata_sdk_submit_external_kv_block_async(req, ctypes.byref(args)))
            if rc != 0:
                self._unpark(key)
            return rc, held
        args = ReadArgs()
        args.struct_size = ctypes.sizeof(ReadArgs)
        args.version = 1
        args.kv_block_index = int(block)
        args.layer_count = len(slices)
        args.layers = slices
        args.bytes_per_layer_slice = int(payload)
        args.sub_stripe_bytes = 0
        args.complete_fn = self._c_complete
        args.complete_user = ctypes.c_void_p(key)
        args.file_ec_unit_bytes = 0
        held["args"] = args
        rc = int(lib.surdata_sdk_read_kv_block_async(req, ctypes.byref(args)))
        if rc != 0:
            self._unpark(key)
        return rc, held

    def _unpark(self, key):
        with self._cb_lock:
            self._cbs.pop(key, None)

    def _user_key(self, user):
        if user is None:
            return 0
        if isinstance(user, int):
            return user
        return int(getattr(user, "value", 0) or 0)
