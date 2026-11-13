import os
import sys
import threading
import time
import traceback
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from surdata_vllm.dram_dock import DramDock
from surdata_vllm.dram_queue import STATUS_FALLBACK_MISS, STATUS_H2D_IN_FLIGHT
from surdata_vllm.kvb import KvGeometry
from surdata_vllm.layout import LayoutError, classify_kv_shape, require_deepseek_mla
from surdata_vllm.sdk_bridge import SdkBridge, SdkError, namespace_model_id
from surdata_vllm.streams import WorkerRuntime, _Block


class DockTests(unittest.TestCase):
    def test_miss_stays_in_the_preload_queue(self):
        dock = DramDock()
        dock.enqueue("m", [1, 2, 3])
        dock.note_index("m", 0)
        self.assertEqual(dock.pipeline.get("m").status, STATUS_FALLBACK_MISS)
        self.assertEqual(dock.matched_tokens("m", 0), (0, False))

    def test_pending_then_preloaded_hit(self):
        dock = DramDock()
        generation = dock.enqueue("r", [0] * 4096)
        self.assertEqual(dock.matched_tokens("r", 1024), (None, False))
        dock.note_index("r", 4096)
        dock.pipeline.mark_preloaded("r", generation, 4096)
        self.assertEqual(dock.matched_tokens("r", 1024), (3072, True))

    def test_h2d_is_not_reported_again(self):
        dock = DramDock()
        generation = dock.enqueue("r", [0] * 32)
        dock.note_index("r", 32)
        dock.pipeline.mark_preloaded("r", generation, 32)
        dock.pipeline.mark_h2d("r", generation)
        self.assertEqual(dock.pipeline.get("r").status, STATUS_H2D_IN_FLIGHT)
        self.assertEqual(dock.matched_tokens("r", 0), (None, False))

    def test_block_zero_is_kept(self):
        dock = DramDock()
        generation = dock.enqueue("r", [0] * 32)
        dock.note_index("r", 32)
        dock.pipeline.mark_preloaded("r", generation, 32)
        self.assertTrue(dock.bind_blocks("r", [0, 3], 32))
        loads = dock.take_loads()
        self.assertEqual(loads[0]["block_ids"], [0, 3])

    def test_dock_does_not_import_lmcache(self):
        path = os.path.join(os.path.dirname(__file__), "..", "surdata_vllm", "dram_dock.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read().lower()
        self.assertNotIn("import lmcache", text)
        self.assertNotIn("surdata-lmcache", text)
        self.assertNotIn("sys.path", text)

    def test_deepseek_shape(self):
        self.assertEqual(require_deepseek_mla((100, 256, 576)), "mla")
        self.assertEqual(require_deepseek_mla((25890, 1, 16, 576)), "mla")
        self.assertEqual(classify_kv_shape((10, 2, 16, 8, 128)), "mha")
        with self.assertRaises(LayoutError):
            require_deepseek_mla((10, 2, 16, 8, 128))


class _Fn:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.owner.calls.append(self.name)
        if self.name == "surdata_sdk_namespace_open":
            self.owner.opened = args[0]
            args[3]._obj.value = 1
        if self.name == "surdata_sdk_connect_storage_from_env":
            delay = self.owner.connect_delay_s
            if delay > 0:
                time.sleep(delay)
            return self.owner.connect_rc
        return 0


class _Lib:
    def __init__(self, connect_rc=0):
        self.calls = []
        self.opened = None
        self.connect_rc = connect_rc
        self.connect_delay_s = 0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fn = _Fn(name, self)
        setattr(self, name, fn)
        return fn


class NamespaceTests(unittest.TestCase):
    def test_model_path_keeps_only_the_last_component(self):
        path = "/root/models/deepseek-ai/DeepSeek-V2-Lite-Chat-AWQ"
        self.assertEqual(namespace_model_id(path), "DeepSeek-V2-Lite-Chat-AWQ")
        self.assertNotIn("/", namespace_model_id(path))
        self.assertEqual(namespace_model_id(""), "m")
        self.assertEqual(namespace_model_id("/"), "m")

    def test_namespace_connects_before_open(self):
        path = "/root/models/deepseek-ai/DeepSeek-V2-Lite-Chat-AWQ"
        model = namespace_model_id(path)
        bridge = SdkBridge(
            {"model": model, "model_version": "v1", "layout_signature": "mla"}
        )
        lib = _Lib()
        bridge._load = lambda: lib
        bridge._namespace()
        self.assertEqual(
            lib.calls[:3],
            [
                "surdata_sdk_init",
                "surdata_sdk_connect_storage_from_env",
                "surdata_sdk_namespace_open",
            ],
        )
        opened = lib.opened.decode() if isinstance(lib.opened, bytes) else lib.opened
        self.assertEqual(opened, "DeepSeek-V2-Lite-Chat-AWQ")
        self.assertNotIn("/", opened)

        failed = SdkBridge(
            {"model": model, "model_version": "v1", "layout_signature": "mla"}
        )
        bad = _Lib(connect_rc=-1)
        failed._load = lambda: bad
        with self.assertRaises(SdkError):
            failed._namespace()
        self.assertNotIn("surdata_sdk_namespace_open", bad.calls)

    def test_two_threads_connect_once(self):
        bridge = SdkBridge(
            {"model": "m", "model_version": "v1", "layout_signature": "mla"}
        )
        lib = _Lib()
        lib.connect_delay_s = 0.05
        bridge._load = lambda: lib
        started = threading.Barrier(2)
        results = []
        errors = []

        def run():
            started.wait()
            try:
                results.append(bridge._namespace())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run), threading.Thread(target=run)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1])
        self.assertEqual(lib.calls.count("surdata_sdk_connect_storage_from_env"), 1)
        self.assertEqual(lib.calls.count("surdata_sdk_namespace_open"), 1)


class _Slot:
    def __init__(self):
        self.buf = bytearray(64)
        self.nbytes = 64


class _Native:
    syncs = 0

    def poll(self, limit=16):
        del limit
        return []


class _SaveFn:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.owner.dispatch(self.name, args)


class _SaveLib:
    def __init__(self):
        self.creates = 0
        self.destroys = 0
        self.publish = []
        self.pub_count = []
        self.old_len = []
        self.commits = []
        self.hash_rc = 0
        self.commit_rc = 0
        self.wave_rc = 0
        self.waves = []
        self.blocks = []
        self.lookup_blocks = 0
        self.lookup_rc = 0
        self.matched_tokens_override = None
        self.bind_rc = 0
        self.binds = []
        self.events = []
        self.hit_destroys = 0
        self._next_hit = 1
        self._hit_blocks = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fn = _SaveFn(name, self)
        setattr(self, name, fn)
        return fn

    def dispatch(self, name, args):
        if name == "surdata_sdk_namespace_open":
            args[3]._obj.value = 1
            return 0
        if name == "surdata_sdk_request_create":
            self.creates += 1
            args[2]._obj.value = self.creates
            return 0
        if name == "surdata_sdk_request_destroy":
            self.destroys += 1
            return 0
        if name == "surdata_sdk_prefix_hashes_from_tokens":
            block = max(1, int(args[2]))
            args[5]._obj.value = int(args[1]) // block
            return 0
        if name == "surdata_sdk_request_set_prefix_hashes":
            return self.hash_rc
        if name == "surdata_sdk_get_metric":
            view = args[0]._obj
            view.align = 8
            view.file_ec_unit = 8
            return 0
        if name == "surdata_sdk_kv_block_extent_bytes":
            def _u64(value):
                if isinstance(value, (bytes, bytearray)):
                    return int.from_bytes(value, "little")
                return int(value)

            payload = _u64(args[0])
            unit = _u64(args[1]) or 1
            return ((payload + unit - 1) // unit) * unit
        if name == "surdata_sdk_lookup_prefix_hit":
            self.events.append("lookup")
            if int(self.lookup_rc) != 0:
                return int(self.lookup_rc)
            self._next_hit += 1
            hid = self._next_hit
            self._hit_blocks[hid] = int(self.lookup_blocks)
            args[3]._obj.value = hid
            return 0
        if name == "surdata_sdk_prefix_hit_get_info":
            hid = int(getattr(args[0], "value", 0) or 0)
            info = args[1]._obj
            blocks = int(self._hit_blocks.get(hid, 0))
            info.found = 1 if blocks > 0 else 0
            info.matched_token_blocks = blocks if blocks > 0 else 0
            if blocks > 0 and self.matched_tokens_override is not None:
                info.matched_tokens = int(self.matched_tokens_override)
            else:
                info.matched_tokens = blocks * 16 if blocks > 0 else 0
            return 0
        if name == "surdata_sdk_request_bind_prefix_hit":
            ful = args[2]
            if ful is None:
                self.binds.append(None)
            else:
                view = ful._obj
                self.binds.append(
                    {
                        "local": int(view.local_token_blocks),
                        "start": int(view.fulfill_token_block_start),
                        "count": int(view.fulfill_token_block_count),
                    }
                )
            self.events.append("bind")
            return int(self.bind_rc)
        if name == "surdata_sdk_prefix_hit_destroy":
            self.hit_destroys += 1
            return 0
        if name == "surdata_sdk_prepare_external_kv_wave_async":
            view = args[1]._obj
            self.events.append("wave")
            self.waves.append(
                (int(view.logical_offset), int(view.wave_kvb_count), int(view.extent_bytes))
            )
            status = int(self.wave_rc)
            total = 0 if status != 0 else int(view.wave_kvb_count) * int(view.extent_bytes)
            view.complete_fn(view.complete_user, status, total)
            return 0
        if name == "surdata_sdk_submit_external_kv_block_async":
            view = args[1]._obj
            self.publish.append(int(view.publish_on_complete))
            self.blocks.append(int(view.kv_block_index))
            self.pub_count.append(int(view.publish_token_block_count))
            self.old_len.append(int(view.old_visible_len))
            view.complete_fn(view.complete_user, 0, int(view.kv_block_index))
            return 0
        if name == "surdata_sdk_request_kv_block_extent":
            return 64
        if name == "surdata_sdk_commit_kv_durable_prefix":
            view = args[1]._obj
            frames = [frame.name for frame in traceback.extract_stack()]
            lens = [int(view.logical_lens[i]) for i in range(int(view.token_block_count))]
            self.commits.append(
                {
                    "start": int(view.token_block_start),
                    "count": int(view.token_block_count),
                    "version": int(view.version),
                    "reason": int(view.publish_reason),
                    "physical": int(view.physical_size_bytes),
                    "lens": lens,
                    "thread": threading.current_thread().name,
                    "frames": frames,
                }
            )
            return self.commit_rc
        return 0


def _wait(pred, seconds=2.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class SaveOnceTests(unittest.TestCase):
    def setUp(self):
        self.lib = _SaveLib()
        self.bridge = SdkBridge({"model": "m", "model_version": "v1", "layout_signature": "mla"})
        self.bridge._load = lambda: self.lib
        self.geo = KvGeometry(4, 16, 1, 1, 1)
        self.bridge.geometry = self.geo
        self.bridge.note_tokens("r", [1] * 32)
        self.slot = _Slot()
        self.rt = WorkerRuntime(sdk=self.bridge, native=_Native(), policy="finish_drop")
        self.rt.set_geometry(self.geo)
        self.rt.start()

    def tearDown(self):
        self.rt.shutdown()

    def _save(self, index, on_done):
        return self.bridge.enqueue_save("r", 1, 16, 16, [self.slot], index, on_done)

    def test_one_request_two_full_blocks_commit_once(self):
        held = []

        def on_done(_status):
            held.append(self.bridge.session_locked("r", 1))

        self.assertEqual(self._save(0, on_done), 0)
        self.assertEqual(self._save(1, on_done), 0)
        self.assertEqual(self.lib.creates, 1)
        self.assertEqual(self.lib.publish, [0, 0])
        self.assertEqual(self.lib.pub_count, [0, 0])
        self.assertEqual(self.lib.old_len, [0, 0])
        self.assertEqual(self.bridge._sessions[("r", 1)].hash_count, 2)
        self.assertEqual(held, [False, False])
        self.assertTrue(self.rt.request_finished("r", [], 32, generation=1))
        self.assertTrue(_wait(lambda: len(self.lib.commits) == 1))
        time.sleep(0.15)
        self.assertEqual(len(self.lib.commits), 1)
        commit = self.lib.commits[0]
        self.assertEqual(commit["start"], 0)
        self.assertEqual(commit["count"], 2)
        self.assertEqual(commit["version"], 2)
        self.assertEqual(commit["reason"], 3)
        self.assertEqual(commit["physical"], 128)
        self.assertEqual(commit["lens"], [16, 32])
        self.assertEqual(commit["thread"], "surdata-commit")
        self.assertNotIn("_store_loop", commit["frames"])
        self.assertIn("_commit_loop", commit["frames"])

    def test_skip_stops_the_prefix(self):
        self.assertEqual(self._save(0, lambda _status: None), 0)
        self.rt._drop_block("r", 1, _Block(1, 16, {}, {}, token_index=1))
        self.assertTrue(_wait(lambda: 1 in self.bridge._sessions[("r", 1)].skipped))
        self.assertTrue(self.rt.request_finished("r", [], 16, generation=1))
        self.assertTrue(_wait(lambda: len(self.lib.commits) == 1))
        time.sleep(0.15)
        self.assertEqual(self.lib.commits[0]["count"], 1)
        self.assertEqual(self.lib.commits[0]["start"], 0)

    def test_abandon_does_not_commit(self):
        self.rt._ensure("r", 1, [1] * 32)
        self.assertEqual(self._save(0, lambda _status: None), 0)
        self.assertIn("r", self.bridge._hashes)
        self.rt.cancel("r")
        self.assertTrue(_wait(lambda: self.lib.destroys >= 1 and "r" not in self.bridge._hashes))
        time.sleep(0.15)
        self.assertEqual(self.lib.commits, [])

    def test_commit_stops_at_hash_count(self):
        self.assertEqual(self._save(0, lambda _status: None), 0)
        self.assertEqual(self._save(1, lambda _status: None), 0)
        self.assertEqual(self._save(2, lambda _status: None), 0)
        self.assertEqual(self.bridge._sessions[("r", 1)].hash_count, 2)
        self.assertIn(2, self.bridge._sessions[("r", 1)].durable)
        self.assertTrue(self.rt.request_finished("r", [], 32, generation=1))
        self.assertTrue(_wait(lambda: len(self.lib.commits) == 1))
        time.sleep(0.15)
        self.assertEqual(len(self.lib.commits), 1)
        self.assertEqual(self.lib.commits[0]["count"], 2)
        self.assertEqual(self.lib.commits[0]["lens"], [16, 32])
        self.assertEqual(self.lib.commits[0]["physical"], 128)
        self.assertTrue(self.bridge._sessions[("r", 1)].destroyed)

    def test_invalid_commit_is_not_retried(self):
        self.lib.commit_rc = -1
        self.assertEqual(self._save(0, lambda _status: None), 0)
        self.assertEqual(self._save(1, lambda _status: None), 0)
        self.assertTrue(self.rt.request_finished("r", [], 32, generation=1))
        self.assertTrue(_wait(lambda: len(self.lib.commits) == 1 and self.lib.destroys >= 1))
        time.sleep(0.15)
        self.assertEqual(len(self.lib.commits), 1)
        sess = self.bridge._sessions[("r", 1)]
        self.assertTrue(sess.destroyed)
        self.assertTrue(sess.failed)
        self.assertFalse(sess.committed)

    def test_first_block_waves_then_submits(self):
        self.assertEqual(self._save(0, lambda _status: None), 0)
        self.assertEqual(len(self.lib.waves), 1)
        offset, count, extent = self.lib.waves[0]
        self.assertEqual(offset, 0)
        self.assertEqual(count, 64)
        self.assertGreater(extent, 0)
        self.assertEqual(self.lib.publish, [0])
        self.assertEqual(self._save(1, lambda _status: None), 0)
        self.assertEqual(len(self.lib.waves), 1)
        self.assertEqual(self.lib.publish, [0, 0])

    def test_half_window_issues_the_next_wave(self):
        self.assertEqual(self._save(0, lambda _status: None), 0)
        extent = self.lib.waves[0][2]
        half = 64 // 2
        self.assertEqual(self._save(half, lambda _status: None), 0)
        self.assertEqual(len(self.lib.waves), 2)
        self.assertEqual(self.lib.waves[1][0], 64 * extent)
        self.assertEqual(self.lib.waves[1][1], 64)
        self.assertEqual(len(self.lib.publish), 2)

    def test_wave_failure_does_not_submit(self):
        self.lib.wave_rc = -1
        self.assertEqual(self._save(0, lambda _status: None), -1)
        self.assertEqual(self.lib.waves, [(0, 64, self.bridge._full_extent)])
        self.assertEqual(self.lib.publish, [])

    def test_published_prefix_hits_and_suffix_is_the_only_save(self):
        """T1 publishes its written prefix. T2 waits, then saves only the new block."""
        t1 = [7] * 32
        t2 = t1 + [8] * 16
        self.bridge.note_tokens("t1", t1)
        self.assertTrue(self.bridge.prefix_may_arrive("t2", t2))
        self.bridge.note_tokens("t2", t2)
        self.bridge.note_hit_blocks("t2", 1, 2)
        held = []

        def on_done(_status):
            held.append(True)

        self.assertEqual(self.bridge.enqueue_save("t1", 1, 16, 16, [self.slot], 0, on_done), 0)
        self.assertEqual(self.bridge.enqueue_save("t1", 1, 16, 16, [self.slot], 1, on_done), 0)
        self.assertTrue(self.rt.request_finished("t1", [], 32, generation=1))
        self.assertTrue(_wait(lambda: any(c["start"] == 0 and c["count"] == 2 for c in self.lib.commits)))
        self.assertFalse(self.bridge.prefix_may_arrive("t2", t2))
        self.assertEqual(self.lib.binds, [])
        before = len(self.lib.blocks)
        self.lib.lookup_blocks = 2
        self.assertEqual(self.bridge.enqueue_save("t2", 1, 16, 16, [self.slot], 0, on_done), 0)
        self.assertEqual(self.bridge.enqueue_save("t2", 1, 16, 16, [self.slot], 1, on_done), 0)
        self.assertEqual(len(self.lib.blocks), before)
        self.assertEqual(self.bridge.enqueue_save("t2", 1, 16, 16, [self.slot], 2, on_done), 0)
        self.assertEqual(self.lib.blocks[before:], [0])
        self.assertEqual(self.lib.waves[-1][0], 0)
        self.assertEqual(self.lib.binds, [{"local": 2, "start": 2, "count": 0}])
        bind_at = self.lib.events.index("bind")
        self.assertLess(bind_at, self.lib.events.index("wave", bind_at))
        self.assertTrue(self.rt.request_finished("t2", [], 48, generation=1))
        self.assertTrue(_wait(lambda: any(c["start"] == 2 and c["count"] == 1 for c in self.lib.commits)))
        suffix = [c for c in self.lib.commits if c["start"] == 2][-1]
        self.assertEqual(suffix["lens"], [48])
        self.assertEqual(suffix["physical"], 64)
        self.assertTrue(_wait(lambda: self.lib.hit_destroys >= 1))
        self.assertIsNone(self.bridge._sessions[("t2", 1)].parent_hit)

    def test_namespace_256_unit_hit_saves_only_the_suffix(self):
        """matched_tokens is 256 per hash. The vLLM block is 16, so 130 hashes are 2080 tokens."""
        tokens = [3] * (131 * 16)
        self.lib.lookup_blocks = 130
        self.lib.matched_tokens_override = 33280
        hit = self.bridge.lookup_hit("wide", tokens, 16, 1)
        self.assertEqual(hit, 2080)
        self.bridge.note_hit_blocks("wide", 1, hit // 16)
        self.assertEqual(self.bridge.enqueue_save("wide", 1, 16, 16, [self.slot], 129, lambda _status: None), 0)
        before = len(self.lib.blocks)
        self.assertEqual(self.bridge.enqueue_save("wide", 1, 16, 16, [self.slot], 130, lambda _status: None), 0)
        self.assertEqual(self.lib.blocks[before:], [0])
        self.assertEqual(self.lib.binds, [{"local": 130, "start": 130, "count": 0}])
        self.assertTrue(self.rt.request_finished("wide", [], 131 * 16, generation=1))
        self.assertTrue(_wait(lambda: any(c["start"] == 130 and c["count"] == 1 for c in self.lib.commits)))
        suffix = [c for c in self.lib.commits if c["start"] == 130][-1]
        self.assertEqual(suffix["lens"], [131 * 16])
        self.assertEqual(suffix["physical"], 64)

    def test_suffix_save_fails_closed_when_parent_length_differs(self):
        self.bridge.note_tokens("t2", [7] * 32 + [8] * 16)
        self.bridge.note_hit_blocks("t2", 1, 2)
        self.lib.lookup_blocks = 1
        before = len(self.lib.waves)
        self.assertEqual(self.bridge.enqueue_save("t2", 1, 16, 16, [self.slot], 2, lambda _status: None), -1)
        self.assertEqual(len(self.lib.waves), before)
        self.assertEqual(self.lib.blocks, [])
        self.assertEqual(self.lib.binds, [])
        self.assertGreaterEqual(self.lib.destroys, 1)
        self.assertGreaterEqual(self.lib.hit_destroys, 1)
        self.assertIsNone(self.bridge._sessions[("t2", 1)].sdk_req)

    def test_enqueue_on_closed_session_returns_invalid(self):
        sess = self.bridge._session("r", 1)
        with sess.lock:
            sess.destroyed = True
        self.assertEqual(self._save(0, lambda _status: None), -1)
        self.assertEqual(self.lib.creates, 0)
        with sess.lock:
            sess.destroyed = False
            sess.committing = True
        self.assertEqual(self._save(0, lambda _status: None), -1)
        self.assertEqual(self.lib.creates, 0)


if __name__ == "__main__":
    unittest.main()
