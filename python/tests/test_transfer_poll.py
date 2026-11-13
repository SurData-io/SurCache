"""Two threads call the real poll binding against a stub library."""

import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _install_vllm_stub():
    if "vllm" in sys.modules and hasattr(sys.modules["vllm"], "__file__"):
        return

    def mod(name):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

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


_install_vllm_stub()

from surdata_vllm.connector import SurDataKVConnector, _NativeAdapter, _flatten_blocks  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: E402

_STUB_C = r"""
#include <stdint.h>
int surdata_transfer_init(void) { return 0; }
int surdata_transfer_cuda_ready(void) { return 0; }
void *surdata_transfer_event_create(void) { return 0; }
int surdata_transfer_event_query(void *event) { (void)event; return 0; }
void surdata_transfer_event_record(void *event, void *stream) {
    (void)event; (void)stream;
}
uint32_t surdata_transfer_poll(void *out, uint32_t cap) {
    (void)out; (void)cap; return 0;
}
int surdata_transfer_submit_device(void) { return 0; }
int surdata_transfer_submit_h2d(void) { return 0; }
int surdata_transfer_submit_d2h(void) { return 0; }
"""


class FlattenBlockTests(unittest.TestCase):
    def test_get_block_ids_before_iter(self):
        class Blocks:
            def __iter__(self):
                raise TypeError("KVCacheBlocks")

            def get_block_ids(self):
                return ([4, 5],)

        self.assertEqual(_flatten_blocks(Blocks()), [4, 5])
        self.assertEqual(_flatten_blocks([1, 2]), [1, 2])
        self.assertEqual(_flatten_blocks(([7], [8])), [7, 8])


class PollRaceTests(unittest.TestCase):
    def test_two_threads_poll_the_same_prototype(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "stub.c")
            so = os.path.join(tmp, "libstub.so")
            with open(src, "w", encoding="utf-8") as handle:
                handle.write(_STUB_C)
            subprocess.check_call(["cc", "-shared", "-fPIC", "-o", so, src])
            previous = os.environ.get("SURDATA_TRANSFER_LIB")
            os.environ["SURDATA_TRANSFER_LIB"] = so
            try:
                adapter = _NativeAdapter()
                errors = []

                def worker():
                    try:
                        for _ in range(400):
                            self.assertEqual(adapter.poll(32), [])
                            threading.Event().wait(0)
                    except Exception as exc:
                        errors.append(exc)

                sys.setswitchinterval(1e-6)
                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(errors, [])
            finally:
                if previous is None:
                    os.environ.pop("SURDATA_TRANSFER_LIB", None)
                else:
                    os.environ["SURDATA_TRANSFER_LIB"] = previous
                sys.setswitchinterval(0.005)


class RequestFinishedTests(unittest.TestCase):
    def test_request_finished_unpacks_as_delay_and_params(self):
        class Request:
            request_id = "r"
            num_tokens = 16

        connector = SurDataKVConnector(object(), KVConnectorRole.SCHEDULER)
        try:
            result = connector.request_finished(Request(), [0, 3])
            delay, params = result
            self.assertEqual(result, (True, None))
            self.assertIs(delay, True)
            self.assertIsNone(params)
        finally:
            connector.shutdown()

    def test_bind_sets_connector_metadata_for_layer_hooks(self):
        connector = SurDataKVConnector(object(), KVConnectorRole.SCHEDULER)
        try:
            first = object()
            self.assertFalse(connector.has_connector_metadata())
            connector.bind_connector_metadata(first)
            self.assertTrue(connector.has_connector_metadata())
            self.assertIs(connector._connector_metadata, first)
            connector.bind_connector_metadata(None)
            self.assertFalse(connector.has_connector_metadata())
            self.assertIsNone(connector._connector_metadata)
            second = object()
            connector.bind_connector_metadata(second)
            self.assertIs(connector._connector_metadata, second)
            connector.clear_connector_metadata()
            self.assertFalse(connector.has_connector_metadata())
            self.assertIsNone(connector._connector_metadata)
        finally:
            connector.shutdown()
