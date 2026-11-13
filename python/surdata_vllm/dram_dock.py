"""Optional queue adapter. The direct connector does not import this module.

The queue is the in-process DramPipelineQueue. This file does not import
the LMCache fork.
"""

from surdata_vllm.dram_queue import (
    STATUS_FALLBACK_MISS,
    STATUS_H2D_IN_FLIGHT,
    STATUS_INIT,
    STATUS_L1_READING,
    STATUS_LOOKUP_PENDING,
    STATUS_PRELOADED,
    DramPipelineQueue,
)

_PENDING = (STATUS_LOOKUP_PENDING, STATUS_INIT, STATUS_L1_READING)


class DramDock:
    def __init__(self, pipeline=None, lookup_client=None, budget=None, allocator=None, native=None):
        self.pipeline = pipeline if pipeline is not None else DramPipelineQueue()
        self.lookup_client = lookup_client
        self.budget = budget
        self.allocator = allocator
        self.native = native
        self._seq = 1
        self._save_jobs = {}
        self._inflight = {}
        self._deferred = {}

    def enqueue(self, req_id: str, token_ids: list) -> int:
        req_id = str(req_id)
        tokens = list(token_ids)
        if self.lookup_client is not None:
            self.lookup_client.lookup(tokens, req_id)
            return 0
        return self.pipeline.enqueue_lookup(req_id, {"token_ids": tokens}, prompt_tokens=len(tokens))

    def note_index(self, req_id: str, hit_tokens: int) -> bool:
        entry = self.pipeline.get(str(req_id))
        generation = 0 if entry is None else int(entry.generation)
        return self.pipeline.leave_preload_record(str(req_id), int(hit_tokens), generation)

    def matched_tokens(self, req_id: str, computed: int, num_tokens=None):
        req_id = str(req_id)
        entry = self.pipeline.get(req_id)
        if entry is None or entry.status in _PENDING:
            return None, False
        if entry.status == STATUS_FALLBACK_MISS:
            return 0, False
        if entry.status == STATUS_H2D_IN_FLIGHT:
            return None, False
        if entry.status != STATUS_PRELOADED:
            return None, False
        ready = int(entry.actual_tokens)
        count = max(0, ready - int(computed))
        if num_tokens is not None and ready >= int(num_tokens) and count > 0:
            count -= 1
        if count <= 0:
            return 0, False
        return count, True

    def bind_blocks(self, req_id: str, block_ids: list, external_tokens: int) -> bool:
        if int(external_tokens) <= 0:
            return False
        ids = [int(block_id) for block_id in block_ids if block_id is not None]
        self._deferred[str(req_id)] = {"block_ids": ids, "tokens": int(external_tokens)}
        return True

    def take_loads(self) -> list:
        loads = []
        for req_id, payload in list(self._deferred.items()):
            loads.append({"req_id": req_id, **payload})
            del self._deferred[req_id]
        return loads

    def inflight(self, req_id: str) -> int:
        return int(self._inflight.get(str(req_id), 0))
