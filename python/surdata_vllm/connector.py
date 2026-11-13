"""vLLM v0.33 connector. Scheduling only posts work. Six worker threads move KV."""

import ctypes
import os
import sys
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from surdata_vllm.config import (
    PRELOAD_POOL_DEFAULT,
    STORE_POOL_DEFAULT,
    load_direct_save_mode,
    load_pool_bytes,
    load_settings,
    load_store_full_policy,
)
from surdata_vllm.kvb import KvGeometry
from surdata_vllm.layout import classify_kv_shape, require_deepseek_mla
from surdata_vllm.sdk_bridge import SdkBridge, namespace_model_id
from surdata_vllm.streams import SchedulerFront, WorkerRuntime, open_scheduler_board, open_worker_board


def _is_scheduler(role) -> bool:
    name = getattr(role, "name", str(role)).lower()
    return name == "scheduler" or role == KVConnectorRole.SCHEDULER


def _flatten_blocks(raw) -> list:
    if raw is None:
        return []
    getter = getattr(raw, "get_block_ids", None)
    if callable(getter):
        raw = getter()
        if raw is None:
            return []
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        return [int(block) for group in raw for block in group]
    try:
        return [int(block) for block in raw]
    except TypeError:
        return []


def _scheduled_count(scheduler_output, req_id: str) -> int:
    """Tokens this step will compute for ``req_id``. Missing table is zero."""
    table = getattr(scheduler_output, "num_scheduled_tokens", None)
    if table is None:
        return 0
    if isinstance(table, dict):
        return int(table.get(req_id, 0) or 0)
    try:
        return int(table[req_id])
    except (KeyError, TypeError, IndexError):
        return 0


def _num_tokens(request) -> int:
    num = getattr(request, "num_tokens", None)
    if num:
        return int(num)
    tokens = getattr(request, "all_token_ids", None) or getattr(request, "prompt_token_ids", None) or []
    return len(tokens)


class TransferView:
    def __init__(self, sending, finished, failed):
        self.finished_sending = set(sending)
        self.finished_recving = set(finished)
        self.failed_recving = set(failed)


def _as_transfer_results(sending, finished, failed):
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )
    except Exception:
        return TransferView(sending, finished, failed)
    try:
        return KVConnectorTransferResults(
            finished_sending=set(sending),
            finished_recving=set(finished),
            failed_recving=set(failed),
        )
    except TypeError:
        return TransferView(sending, finished, failed)


class _Job(ctypes.Structure):
    """surdata_job_id_t. One type for every thread."""

    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("worker_epoch", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
    ]


class _Span(ctypes.Structure):
    """surdata_copy_span_t."""

    _fields_ = [
        ("src", ctypes.c_void_p),
        ("dst", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("layer", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("producer", ctypes.c_void_p),
    ]


class _Completion(ctypes.Structure):
    """surdata_completion_t. job_id is the first three uint64 fields."""

    _fields_ = [
        ("engine_epoch", ctypes.c_uint64),
        ("worker_epoch", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("request_epoch", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("valid_bytes", ctypes.c_uint64),
        ("direction", ctypes.c_int),
        ("status", ctypes.c_int),
    ]


class _NativeAdapter:
    """ctypes transfer library. Tests pass their own native double instead."""

    def __init__(self):
        path = os.environ.get("SURDATA_TRANSFER_LIB", "libsurdata_transfer.so")
        self._lib = ctypes.CDLL(path)
        lib = self._lib
        lib.surdata_transfer_init.restype = ctypes.c_int
        lib.surdata_transfer_cuda_ready.restype = ctypes.c_int
        lib.surdata_transfer_event_create.restype = ctypes.c_void_p
        lib.surdata_transfer_event_query.argtypes = [ctypes.c_void_p]
        lib.surdata_transfer_event_query.restype = ctypes.c_int
        lib.surdata_transfer_event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.surdata_transfer_event_record.restype = ctypes.c_int
        destroy = getattr(lib, "surdata_transfer_event_destroy", None)
        self._event_destroy = destroy
        if destroy is not None:
            destroy.argtypes = [ctypes.c_void_p]
        lib.surdata_transfer_poll.argtypes = [ctypes.POINTER(_Completion), ctypes.c_uint32]
        lib.surdata_transfer_poll.restype = ctypes.c_uint32
        lib.surdata_transfer_submit_device.argtypes = [
            _Job,
            ctypes.POINTER(_Span),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.surdata_transfer_submit_device.restype = ctypes.c_int
        lib.surdata_transfer_submit_h2d.argtypes = [
            _Job,
            ctypes.POINTER(_Span),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.surdata_transfer_submit_h2d.restype = ctypes.c_int
        lib.surdata_transfer_submit_d2h.argtypes = [
            _Job,
            ctypes.POINTER(_Span),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        lib.surdata_transfer_submit_d2h.restype = ctypes.c_int
        cancel = getattr(lib, "surdata_transfer_request_cancel", None)
        self._request_cancel = cancel
        if cancel is not None:
            cancel.argtypes = [_Job]
            cancel.restype = None
        self._host_pin = getattr(lib, "surdata_transfer_host_pin", None)
        self._host_unpin = getattr(lib, "surdata_transfer_host_unpin", None)
        self._hostmem_alloc = getattr(lib, "surdata_hostmem_alloc", None)
        self._hostmem_free = getattr(lib, "surdata_hostmem_free", None)
        self._copy_async_check = getattr(lib, "surdata_transfer_copy_async_check", None)
        self._memcpy_async = getattr(lib, "surdata_transfer_memcpy_async", None)
        if self._host_pin is not None:
            self._host_pin.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._host_pin.restype = ctypes.c_int
        if self._host_unpin is not None:
            self._host_unpin.argtypes = [ctypes.c_void_p]
            self._host_unpin.restype = ctypes.c_int
        if self._hostmem_alloc is not None:
            self._hostmem_alloc.argtypes = [
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            self._hostmem_alloc.restype = ctypes.c_int
        if self._hostmem_free is not None:
            self._hostmem_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._hostmem_free.restype = None
        if self._copy_async_check is not None:
            self._copy_async_check.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self._copy_async_check.restype = ctypes.c_int
        if self._memcpy_async is not None:
            self._memcpy_async.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._memcpy_async.restype = ctypes.c_int
        self.syncs = 0
        self.call_threads = []
        if lib.surdata_transfer_init() != 0:
            raise RuntimeError("surdata_transfer_init failed")

    def event_create(self):
        return self._lib.surdata_transfer_event_create()

    def event_record(self, event, stream=None) -> int:
        ptr = ctypes_stream(stream)
        return int(self._lib.surdata_transfer_event_record(event, ptr))

    def event_destroy(self, event) -> None:
        fn = getattr(self, "_event_destroy", None)
        if fn is not None:
            fn(event)

    def event_query(self, event) -> int:
        return int(self._lib.surdata_transfer_event_query(event))

    def cuda_ready(self) -> bool:
        return bool(self._lib.surdata_transfer_cuda_ready())

    def memcpy_async(self, dst, src, nbytes, stream, kind) -> int:
        if self._memcpy_async is None:
            raise RuntimeError("surdata_transfer_memcpy_async missing")
        return int(
            self._memcpy_async(
                ctypes.c_void_p(int(dst)),
                ctypes.c_void_p(int(src)),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_void_p(int(stream)),
                ctypes.c_int(int(kind)),
            )
        )

    def host_pin(self, ptr, nbytes) -> int:
        if self._host_pin is None:
            raise RuntimeError("surdata_transfer_host_pin missing")
        if not self.cuda_ready():
            self._lib.surdata_transfer_init()
        return int(self._host_pin(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(nbytes))))

    def hostmem_alloc(self, nbytes):
        if self._hostmem_alloc is None:
            raise RuntimeError("surdata_hostmem_alloc missing")
        out = ctypes.c_void_p()
        mapped = ctypes.c_size_t()
        rc = int(
            self._hostmem_alloc(
                ctypes.c_size_t(int(nbytes)), ctypes.byref(out), ctypes.byref(mapped)
            )
        )
        if rc != 0 or not out.value:
            raise RuntimeError("surdata_hostmem_alloc failed rc=%s bytes=%s" % (rc, int(nbytes)))
        return int(out.value), int(mapped.value)

    def hostmem_free(self, ptr, mapped_bytes) -> None:
        if self._hostmem_free is None:
            raise RuntimeError("surdata_hostmem_free missing")
        if not ptr:
            return
        self._hostmem_free(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(mapped_bytes)))

    def copy_async_check(self, ptr, nbytes, direction) -> int:
        if self._copy_async_check is None:
            raise RuntimeError("surdata_transfer_copy_async_check missing")
        if not self.cuda_ready():
            self._lib.surdata_transfer_init()
        return int(
            self._copy_async_check(
                ctypes.c_void_p(int(ptr)),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_int(int(direction)),
            )
        )

    def host_unpin(self, ptr) -> int:
        if self._host_unpin is None:
            return -1
        return int(self._host_unpin(ctypes.c_void_p(int(ptr))))

    def request_cancel(self, job) -> None:
        cancel = getattr(self, "_request_cancel", None)
        if cancel is None or not isinstance(job, dict):
            return
        ident = _Job(int(job["engine_epoch"]), int(job["worker_epoch"]), int(job["sequence"]))
        cancel(ident)

    def submit_h2d(self, job, spans, epoch, generation):
        import threading

        self.call_threads.append(threading.get_ident())
        return self._submit(job, spans, epoch, generation, 1, None)

    def submit_d2h(self, job, spans, epoch, generation, producers):
        import threading

        self.call_threads.append(threading.get_ident())
        return self._submit(job, spans, epoch, generation, 2, producers)

    def poll(self, limit=16):
        out = (_Completion * int(limit))()
        n = int(self._lib.surdata_transfer_poll(out, int(limit)))
        rows = []
        for i in range(n):
            rows.append(
                {
                    "sequence": int(out[i].sequence),
                    "status": int(out[i].status),
                    "direction": int(out[i].direction),
                    "generation": int(out[i].generation),
                }
            )
        return rows

    def _submit(self, job, spans, epoch, generation, direction, producers):
        buf = (_Span * len(spans))()
        for i, span in enumerate(spans):
            buf[i].src = _ptr(span.get("src"))
            buf[i].dst = _ptr(span.get("dst"))
            buf[i].bytes = int(span.get("bytes") or 0)
            layer = span.get("layer", i)
            buf[i].layer = int(layer) if isinstance(layer, int) else i
            producer = None if not producers else producers[i] if i < len(producers) else None
            buf[i].producer = _ptr(producer)
        ident = _Job(int(job["engine_epoch"]), int(job["worker_epoch"]), int(job["sequence"]))
        if self.cuda_ready():
            return int(
                self._lib.surdata_transfer_submit_device(
                    ident, buf, len(spans), int(epoch), int(generation), int(direction)
                )
            )
        if direction == 1:
            return int(
                self._lib.surdata_transfer_submit_h2d(
                    ident, buf, len(spans), int(epoch), int(generation)
                )
            )
        return int(
            self._lib.surdata_transfer_submit_d2h(
                ident, buf, len(spans), int(epoch), int(generation), None
            )
        )


def ctypes_stream(stream):
    import ctypes

    if not stream:
        return ctypes.c_void_p()
    return ctypes.c_void_p(int(stream))


def _ptr(value):
    import ctypes

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, ctypes.c_void_p):
        if not value.value:
            return None
        return int(value.value)
    if isinstance(value, (bytes, bytearray)):
        return ctypes.addressof(ctypes.c_char.from_buffer(value))
    data = getattr(value, "buf", None)
    if data is not None:
        return ctypes.addressof(ctypes.c_char.from_buffer(data))
    return None


class SurDataKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        if kv_cache_config is not None:
            super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        else:
            super().__init__(vllm_config=vllm_config, role=role)
        self._scheduler = _is_scheduler(role)
        model = ""
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            model = str(getattr(model_config, "model", "") or "")
        self._expect_mla = "deepseek" in model.lower()
        cache = getattr(vllm_config, "cache_config", None)
        self._block_size = int(getattr(cache, "block_size", 16) or 16)
        self._settings = load_settings(
            {
                "block_tokens": self._block_size,
                "bytes_per_token": 1,
                "layout_signature": "mla" if self._expect_mla else "kv",
                "partial_tail_restore": False,
                "model": namespace_model_id(model),
            }
        )
        if self._scheduler:
            self._board = open_scheduler_board()
            self._front = SchedulerFront(self._board)
            self._worker = None
            self._native = None
        else:
            self._board = open_worker_board()
            self._native = _NativeAdapter()
            self._worker = WorkerRuntime(
                sdk=SdkBridge(self._settings),
                native=self._native,
                policy=load_store_full_policy(),
                save_mode=load_direct_save_mode(),
                board=self._board,
                preload_bytes=load_pool_bytes("kv.preload_pool_bytes", PRELOAD_POOL_DEFAULT),
                store_bytes=load_pool_bytes("kv.store_pool_bytes", STORE_POOL_DEFAULT),
            )
            self._worker.start()

    def register_kv_caches(self, kv_caches):
        shapes = []
        itemsize = 0
        for tensor in kv_caches.values():
            shape = tuple(int(dim) for dim in tensor.shape)
            shapes.append(shape)
            itemsize = int(tensor.element_size())
            if self._expect_mla:
                require_deepseek_mla(shape)
            else:
                classify_kv_shape(shape)
        if self._worker is None or not shapes:
            return
        self._worker.set_geometry(KvGeometry.from_layers(shapes, itemsize))
        bases = {}
        for name, tensor in kv_caches.items():
            ptr = getattr(tensor, "data_ptr", None)
            if callable(ptr):
                bases[str(name)] = int(ptr())
        if bases:
            self._worker.set_layer_bases(bases)

    def bind_connector_metadata(self, metadata) -> None:
        super().bind_connector_metadata(metadata)
        if self._worker is not None and metadata is not None:
            self._worker.ingest(metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._worker is not None:
            self._worker.clear_published()

    def start_load_kv(self, forward_context, **kwargs) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        if self._worker is None:
            return
        # end records one event in wait_for_save. vLLM still calls this hook.
        if self._worker.save_mode == "end":
            return
        trace = os.environ.get("SURDATA_DECODE_TRACE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        started = time.perf_counter() if trace else 0.0
        create_ms = 0.0
        record_ms = 0.0
        try:
            shape = tuple(int(dim) for dim in getattr(kv_layer, "shape", ()) or ())
            if shape:
                if self._expect_mla:
                    require_deepseek_mla(shape)
                else:
                    classify_kv_shape(shape)
            create_started = time.perf_counter() if trace else 0.0
            event = self._native.event_create()
            if trace:
                create_ms = (time.perf_counter() - create_started) * 1000.0
            stream = None
            try:
                import torch

                if torch.cuda.is_available():
                    current = torch.cuda.current_stream()
                    self._worker.note_compute_stream(current)
                    stream = current.cuda_stream
            except Exception:
                stream = None
            record_started = time.perf_counter() if trace else 0.0
            rc = int(self._native.event_record(event, stream))
            if trace:
                record_ms = (time.perf_counter() - record_started) * 1000.0
            if rc != 0:
                self._native.event_destroy(event)
                print(
                    "SURDATA_EVENT record failed rc=%s layer=%s" % (rc, layer_name),
                    file=sys.stderr,
                    flush=True,
                )
                return
            self._worker.save_layer_event(str(layer_name), int(kv_layer.data_ptr()), event)
        finally:
            if trace:
                save_ms = (time.perf_counter() - started) * 1000.0
                self._worker.note_save_layer_trace(create_ms, record_ms, save_ms)

    def wait_for_save(self):
        if self._worker is None:
            return None
        self._worker.note_compute_stream()
        self._worker.wait_for_save()
        return None

    def get_finished(self, finished_req_ids):
        if self._worker is None:
            return None
        return self._worker.get_finished()

    def get_transfer_results(self, finished_req_ids=None):
        if self._worker is None:
            return _as_transfer_results(set(), set(), set())
        got = self._worker.transfer_results(finished_req_ids)
        return _as_transfer_results(got.finished_sending, got.finished_recving, got.failed_recving)

    def get_block_ids_with_load_errors(self):
        return []

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()
        if self._board is not None:
            self._board.close()

    def on_new_request(self, request) -> None:
        if not self._scheduler:
            return
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        if not tokens:
            return
        self._front.on_new_request(str(request.request_id), tokens)

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        if not self._scheduler:
            return None, False
        return self._front.matched_tokens(str(request.request_id), int(num_computed_tokens))

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        if not self._scheduler:
            return
        self._front.note_alloc(str(request.request_id), _flatten_blocks(blocks), int(num_external_tokens))

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = self._front.build_meta()
        self._attach_save_hints(scheduler_output, meta)
        return meta

    def request_finished(self, request, block_ids):
        if self._scheduler:
            ids = _flatten_blocks(block_ids)
            num = _num_tokens(request)
            req_id = str(request.request_id)
            self._front.note_save(req_id, ids, num)
            self._front.note_finished(req_id, ids, num)
        return True, None

    def handle_preemptions(self, kv_connector_metadata) -> None:
        ids = []
        if kv_connector_metadata is None:
            return
        raw = getattr(kv_connector_metadata, "preempted", None)
        if raw is None and isinstance(kv_connector_metadata, (list, tuple, set)):
            raw = kv_connector_metadata
        for item in raw or []:
            req_id = item if isinstance(item, str) else getattr(item, "req_id", None) or getattr(item, "request_id", None)
            if req_id is not None:
                ids.append(str(req_id))
        if self._scheduler:
            for req_id in ids:
                self._front.note_preempt(req_id)
        elif self._worker is not None:
            for req_id in ids:
                self._worker.preempt(req_id)

    def register_finished_partial_tail(self, request, block_ids, partial_tail_offloads) -> bool:
        raise RuntimeError("partial tail restore is not supported")

    def _attach_save_hints(self, scheduler_output, meta) -> None:
        """One hint per scheduled request, with the tokens this forward will finish.

        ``num_computed_tokens`` is what HBM already holds. Adding this step's
        ``num_scheduled_tokens`` is the count ``wait_for_save`` may copy after
        the forward. The prompt length is not used: blocks past the forward
        would be copied before they are written. Cached requests append only
        ``new_block_ids`` onto the table kept on the scheduler.
        """
        seen = set()
        for attr in ("new_reqs", "scheduled_new_reqs"):
            for req in getattr(scheduler_output, attr, None) or []:
                req_id = str(getattr(req, "req_id", "") or getattr(req, "request_id", ""))
                if not req_id or req_id in seen:
                    continue
                seen.add(req_id)
                ids = _flatten_blocks(getattr(req, "block_ids", None))
                computed = int(getattr(req, "num_computed_tokens", 0) or 0)
                num = computed + _scheduled_count(scheduler_output, req_id)
                if num <= 0:
                    num = int(getattr(req, "num_tokens", 0) or 0)
                self._front.remember_blocks(req_id, ids, num, True)
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            req_ids = list(getattr(cached, "req_ids", None) or [])
            new_ids = list(getattr(cached, "new_block_ids", None) or [])
            computed_list = list(getattr(cached, "num_computed_tokens", None) or [])
            for idx, raw_id in enumerate(req_ids):
                req_id = str(raw_id)
                if not req_id or req_id in seen:
                    continue
                seen.add(req_id)
                extra_ids = _flatten_blocks(new_ids[idx]) if idx < len(new_ids) else []
                computed = int(computed_list[idx]) if idx < len(computed_list) else 0
                num = computed + _scheduled_count(scheduler_output, req_id)
                self._front.remember_blocks(req_id, extra_ids, num, False)
        extra = self._front.build_meta()
        meta.save_hints.extend(extra.save_hints)
        meta.lookups.extend(extra.lookups)
        meta.allocs.extend(extra.allocs)
        meta.finished.extend(extra.finished)
        meta.preempted.extend(extra.preempted)
