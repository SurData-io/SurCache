"""Scheduler and worker control that does not import vLLM.

Snapshot reads never wait. A positive match still needs an asynchronous H2D.
"""

from surdata_vllm.config import load_settings


class SchedulerCore:
    def __init__(self, sdk, settings: dict):
        self.sdk = sdk
        self.settings = load_settings(settings)
        self._offers = {}
        self._finished = set()

    def on_new_request(self, request_id: int, epoch: int, token_ids: list[int]) -> None:
        self.sdk.preload_enqueue(
            request_id,
            epoch,
            token_ids,
            self.settings["block_tokens"],
            self.settings["bytes_per_token"],
            self.settings["layout_signature"],
            self.settings["shard_id"],
        )

    def matched_tokens(self, request_id: int, epoch: int, num_computed: int):
        snap = self.sdk.preload_snapshot(request_id, epoch)
        status = snap["status"]
        if status == "pending":
            return None, True
        if status == "miss":
            return 0, False
        ready = int(snap["ready_tokens"])
        count = max(0, ready - int(num_computed))
        if count <= 0:
            return 0, False
        lease = self.sdk.try_hold_read(request_id, epoch, int(snap["generation"]))
        if lease is None:
            return None, True
        self._offers[(request_id, epoch)] = lease
        return count, True

    def bind_blocks(self, request_id: int, epoch: int, block_ids: list[int], external_tokens: int):
        key = (request_id, epoch)
        if external_tokens <= 0:
            self._offers.pop(key, None)
            return None
        offer = self._offers.get(key)
        if offer is None:
            raise RuntimeError("missing load offer")
        offer = dict(offer)
        offer["block_ids"] = list(block_ids)
        offer["external_tokens"] = int(external_tokens)
        self._offers[key] = offer
        return offer

    def take_loads(self):
        loads = []
        for key, offer in list(self._offers.items()):
            if "block_ids" not in offer:
                continue
            loads.append({"request_id": key[0], "epoch": key[1], **offer})
            del self._offers[key]
        return loads

    def note_finished(self, request_id: int) -> bool:
        """Hold every GPU block of the request until its saves finish."""
        self._finished.add(request_id)
        return True


class WorkerCore:
    def __init__(self, native):
        self.native = native
        self._saves = []
        self._fenced = False

    def save_kv_layer(self, spans: list[dict], request_epoch: int, generation: int, job: dict):
        self._saves.append(
            {
                "spans": spans,
                "request_epoch": request_epoch,
                "generation": generation,
                "job": job,
            }
        )
        self._fenced = True

    def wait_for_save(self):
        if not self._saves:
            return []
        if not self._fenced:
            raise RuntimeError("D2H submitted without a producer fence from save_kv_layer")
        fence = self.native.fence_create()
        self.native.fence_signal(fence)
        submitted = []
        for save in self._saves:
            rc = self.native.submit_d2h(
                save["job"], save["spans"], save["request_epoch"], save["generation"], fence
            )
            if rc != 0:
                raise RuntimeError(f"submit_d2h rc={rc}")
            submitted.append(save["job"])
        self._saves = []
        self._fenced = False
        return submitted

    def start_load(self, job: dict, spans: list[dict], request_epoch: int, generation: int):
        rc = self.native.submit_h2d(job, spans, request_epoch, generation)
        if rc != 0:
            raise RuntimeError(f"submit_h2d rc={rc}")

    def poll(self, limit: int = 16):
        return self.native.poll(limit)
