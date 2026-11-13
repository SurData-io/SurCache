"""Worker-owned preload queue, indexed by the status-board slot.

``items[slot]`` is the row. ``_order`` stores those slot numbers in admission
order. The scheduler only appends a slot to ``_wait``. One worker thread
moves a slot onto ``_order`` once the preload pool can hold the prompt.
There is no lookup window and no queue mutex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import time


class QueueStatus(IntEnum):
    """In-process preload queue. Integers stay clear of the status-board phases."""

    LOOKUP_PENDING = 16
    INIT = 17
    L1_READING = 18
    PRELOADED = 19
    H2D_IN_FLIGHT = 20
    DONE = 21
    FALLBACK_MISS = 22


STATUS_LOOKUP_PENDING = QueueStatus.LOOKUP_PENDING
STATUS_INIT = QueueStatus.INIT
STATUS_L1_READING = QueueStatus.L1_READING
STATUS_PRELOADED = QueueStatus.PRELOADED
STATUS_H2D_IN_FLIGHT = QueueStatus.H2D_IN_FLIGHT
STATUS_DONE = QueueStatus.DONE
STATUS_FALLBACK_MISS = QueueStatus.FALLBACK_MISS

_SLOTS = 1024


@dataclass
class DramPipelineEntry:
    req_id: str
    slot: int = -1
    lookup_hit_tokens: int = 0
    actual_tokens: int = 0
    prompt_tokens: int = 0
    status: QueueStatus = STATUS_LOOKUP_PENDING
    order: int = 0
    generation: int = 1
    enqueued_mono: float = field(default_factory=time.monotonic)
    lookup_input: object = None
    preload_job: object = None
    lookup_claimed: bool = False
    preload_claimed: bool = False
    lookup_retries: int = 0
    retry_after: float = 0.0
    ticket: int = 0
    reserved: object = None


class DramPipelineQueue:
    def __init__(self, lookup_limit: int = 0) -> None:
        del lookup_limit
        self.items: list[DramPipelineEntry | None] = [None] * _SLOTS
        self._order: list[int] = []
        self._wait: list[int] = []
        self._slot_of: dict[str, int] = {}
        self._next_order = 0
        self._next_auto = 0
        self.enqueued = 0
        self.preloaded = 0
        self.fallback = 0
        self._closed: set[str] = set()
        self._inflight: dict[str, int] = {}
        self._need = None
        self._free = None

    def set_capacity(self, need, free) -> None:
        """``need(entry)`` and ``free()`` are bytes. Either may be absent."""
        self._need = need
        self._free = free

    def _bytes(self, entry: DramPipelineEntry) -> int:
        fn = self._need
        if not callable(fn):
            return 0
        try:
            return max(0, int(fn(entry)))
        except Exception:
            return 0

    def _room_for(self, entry: DramPipelineEntry) -> bool:
        fn = self._free
        if not callable(fn):
            return True
        try:
            return int(fn()) >= self._bytes(entry)
        except Exception:
            return True

    def _find(self, req_id: str) -> DramPipelineEntry | None:
        slot = self._slot_of.get(str(req_id))
        if slot is None:
            return None
        entry = self.items[int(slot)]
        if entry is None or entry.req_id != str(req_id):
            return None
        return entry

    def _alloc_slot(self, requested) -> int:
        if requested is not None:
            slot = int(requested)
            if slot < 0 or slot >= _SLOTS:
                raise IndexError("status slot %s" % slot)
            return slot
        for _ in range(_SLOTS):
            slot = self._next_auto
            self._next_auto = (slot + 1) % _SLOTS
            if self.items[slot] is None:
                return slot
        raise RuntimeError("preload queue slots are full")

    def enqueue_lookup(self, req_id: str, lookup_input, prompt_tokens: int = 0, generation: int | None = None, slot=None) -> int:
        """Record a request on the wait ring. Does not start a lookup."""
        key = str(req_id)
        self._closed.discard(key)
        entry = self._find(key)
        chosen = 1 if generation is None else int(generation)
        if (
            entry is not None
            and entry.status == STATUS_LOOKUP_PENDING
        ):
            entry.generation = chosen
            entry.lookup_input = lookup_input
            entry.lookup_claimed = False
            entry.ticket += 1
            entry.lookup_retries = 0
            if prompt_tokens:
                entry.prompt_tokens = max(0, int(prompt_tokens))
            if entry.slot not in self._wait and entry.slot not in self._order:
                self._wait.append(entry.slot)
            return entry.generation
        if entry is not None:
            self._drop_slot(entry.slot)
        slot_i = self._alloc_slot(slot if slot is not None else (None if entry is None else entry.slot))
        if self.items[slot_i] is not None and self.items[slot_i].req_id != key:
            self._drop_slot(slot_i)
        row = DramPipelineEntry(
            req_id=key,
            slot=slot_i,
            prompt_tokens=max(0, int(prompt_tokens)),
            status=STATUS_LOOKUP_PENDING,
            order=self._next_order,
            generation=chosen,
            lookup_input=lookup_input,
        )
        self._next_order += 1
        self.items[slot_i] = row
        self._slot_of[key] = slot_i
        self._wait.append(slot_i)
        self.enqueued += 1
        return row.generation

    def drop_wait(self, req_id: str) -> None:
        """A request that never received DRAM leaves the wait ring only."""
        entry = self._find(req_id)
        if entry is None:
            return
        if entry.slot in self._wait and entry.status == STATUS_LOOKUP_PENDING and not entry.lookup_claimed:
            self._wait = [slot for slot in self._wait if slot != entry.slot]
            self._drop_slot(entry.slot)

    def _lookup_ready(self, entry: DramPipelineEntry | None, now: float) -> bool:
        return (
            entry is not None
            and entry.status == STATUS_LOOKUP_PENDING
            and not entry.lookup_claimed
            and entry.lookup_input is not None
            and float(entry.retry_after) <= now
        )

    def take_next_lookup(self):
        now = time.monotonic()
        for slot in self._order:
            entry = self.items[slot]
            if not self._lookup_ready(entry, now):
                continue
            entry.lookup_claimed = True
            self._inflight[entry.req_id] = int(entry.ticket)
            return entry.req_id, int(entry.generation), entry.lookup_input
        kept = []
        chosen = None
        for slot in self._wait:
            entry = self.items[slot]
            if not self._lookup_ready(entry, now):
                if entry is not None:
                    kept.append(slot)
                continue
            if chosen is None and self._room_for(entry):
                chosen = slot
                continue
            kept.append(slot)
        self._wait = kept
        if chosen is None:
            return None
        entry = self.items[chosen]
        self._order.append(chosen)
        entry.lookup_claimed = True
        self._inflight[entry.req_id] = int(entry.ticket)
        return entry.req_id, int(entry.generation), entry.lookup_input

    def defer_for_dram(self, req_id: str) -> None:
        """The preload pool cannot hold this prompt yet. Return it to the wait ring."""
        entry = self._find(req_id)
        if entry is None or entry.status != STATUS_LOOKUP_PENDING:
            return
        entry.lookup_claimed = False
        entry.reserved = None
        slot = int(entry.slot)
        self._order = [item for item in self._order if item != slot]
        if slot not in self._wait:
            self._wait.append(slot)

    def note_reserved(self, req_id: str, slots) -> None:
        entry = self._find(req_id)
        if entry is not None:
            entry.reserved = slots

    def take_reserved(self, req_id: str):
        entry = self._find(req_id)
        if entry is None:
            return None
        slots = entry.reserved
        entry.reserved = None
        return slots

    def claim_preload(self, req_id: str) -> bool:
        entry = self._find(req_id)
        if entry is None or entry.status != STATUS_INIT or entry.preload_claimed:
            return False
        entry.preload_claimed = True
        return True

    def release_lookup_window(self, req_id: str) -> bool:
        """Admission is DRAM capacity. There is no window to return."""
        del req_id
        return False

    def requeue_lookup(self, req_id: str, generation: int, delay: float = 0.05) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None or entry.status != STATUS_LOOKUP_PENDING:
            return False
        entry.lookup_claimed = False
        entry.retry_after = time.monotonic() + max(0.0, float(delay))
        return True

    def release_lookup(self, req_id: str, generation: int, delay: float = 0.05) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None or entry.status != STATUS_LOOKUP_PENDING:
            return False
        if int(entry.lookup_retries) >= 1:
            return False
        entry.lookup_retries = 1
        entry.lookup_claimed = False
        entry.retry_after = time.monotonic() + max(0.0, float(delay))
        return True

    def next_lookup_delay(self) -> float:
        now = time.monotonic()
        soon = None
        slots = list(self._order) + list(self._wait)
        for slot in slots:
            entry = self.items[slot]
            if (
                entry is None
                or entry.status != STATUS_LOOKUP_PENDING
                or entry.lookup_claimed
                or entry.lookup_input is None
            ):
                continue
            if slot in self._wait and not self._room_for(entry):
                continue
            left = float(entry.retry_after) - now
            if left <= 0:
                return 0.0
            soon = left if soon is None else min(soon, left)
        return 0.05 if soon is None else soon

    def complete_lookup(self, req_id: str, generation: int, hit_tokens: int, prompt_tokens: int = 0, ticket: int | None = None) -> bool:
        del generation
        hit = max(0, int(hit_tokens))
        key = str(req_id)
        if key in self._closed:
            return False
        entry = self._find(key)
        if entry is None or entry.status != STATUS_LOOKUP_PENDING:
            return False
        stamped = self._inflight.get(key)
        if stamped is not None and int(stamped) != int(entry.ticket):
            return False
        entry.lookup_hit_tokens = hit
        entry.lookup_input = None
        if prompt_tokens:
            entry.prompt_tokens = max(entry.prompt_tokens, int(prompt_tokens))
        if hit <= 0:
            entry.actual_tokens = 0
            entry.preload_job = None
            entry.status = STATUS_FALLBACK_MISS
            self.fallback += 1
        else:
            entry.status = STATUS_INIT
        return True

    def leave_preload_record(self, req_id: str, hit_tokens: int, generation: int, prompt_tokens: int = 0) -> bool:
        return self.complete_lookup(req_id, generation, hit_tokens, prompt_tokens)

    def take_next_preload(self):
        now = time.monotonic()
        for slot in self._order:
            entry = self.items[slot]
            if (
                entry is None
                or entry.status != STATUS_INIT
                or entry.preload_claimed
                or entry.lookup_hit_tokens <= 0
                or float(entry.retry_after) > now
            ):
                continue
            entry.preload_claimed = True
            return entry.req_id, int(entry.generation), entry.lookup_hit_tokens
        return None

    def next_preload_delay(self) -> float:
        now = time.monotonic()
        soon = None
        for slot in self._order:
            entry = self.items[slot]
            if (
                entry is None
                or entry.status != STATUS_INIT
                or entry.preload_claimed
                or entry.lookup_hit_tokens <= 0
            ):
                continue
            left = float(entry.retry_after) - now
            if left <= 0:
                return 0.0
            soon = left if soon is None else min(soon, left)
        return 0.05 if soon is None else soon

    def unclaim_preload(self, req_id: str, generation: int) -> None:
        del generation
        entry = self._find(req_id)
        if entry is None:
            return
        if entry.status == STATUS_INIT:
            entry.preload_claimed = False

    def retry_preload(self, req_id: str, generation: int, delay: float = 0.05) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None or entry.status not in (STATUS_INIT, STATUS_L1_READING):
            return False
        entry.status = STATUS_INIT
        entry.preload_claimed = False
        entry.retry_after = time.monotonic() + max(0.0, float(delay))
        return True

    def mark_fallback(self, req_id: str, generation: int) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None:
            return False
        if entry.status in (STATUS_DONE, STATUS_FALLBACK_MISS):
            return False
        entry.lookup_hit_tokens = 0
        entry.actual_tokens = 0
        entry.status = STATUS_FALLBACK_MISS
        self.fallback += 1
        return True

    def mark_reading(self, req_id: str, generation: int) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None or entry.status != STATUS_INIT:
            return False
        entry.status = STATUS_L1_READING
        return True

    def mark_preloaded(self, req_id: str, generation: int, actual_tokens: int) -> bool:
        del generation
        key = str(req_id)
        if key in self._closed:
            return False
        entry = self._find(key)
        if entry is None:
            return False
        if entry.status in (STATUS_DONE, STATUS_FALLBACK_MISS, STATUS_H2D_IN_FLIGHT, STATUS_PRELOADED):
            return False
        actual = max(0, int(actual_tokens))
        if actual <= 0:
            return False
        entry.actual_tokens = actual
        entry.status = STATUS_PRELOADED
        entry.preload_job = None
        self.preloaded += 1
        return True

    def mark_h2d(self, req_id: str, generation: int) -> None:
        """Assign H2D_IN_FLIGHT when this slot is still the same request."""
        del generation
        entry = self._find(req_id)
        if entry is None:
            return
        if entry.status not in (STATUS_PRELOADED, STATUS_H2D_IN_FLIGHT):
            return
        entry.status = STATUS_H2D_IN_FLIGHT

    def mark_done(self, req_id: str, generation: int) -> bool:
        del generation
        entry = self._find(req_id)
        if entry is None:
            return False
        entry.status = STATUS_DONE
        return True

    def drop_loaded(self, req_id: str, generation: int) -> bool:
        """Drop a row that has reached HBM. The request stays reusable."""
        del generation
        entry = self._find(req_id)
        if entry is None:
            return False
        self._drop_slot(entry.slot)
        return True

    def insert_preloaded(self, req_id: str, generation: int, actual_tokens: int, slot=None) -> None:
        """Park a computed prefix. Does not take a DRAM admission place."""
        key = str(req_id)
        tokens = max(0, int(actual_tokens))
        self._closed.discard(key)
        entry = self._find(key)
        if entry is not None:
            self._drop_slot(entry.slot)
        slot_i = self._alloc_slot(slot)
        if self.items[slot_i] is not None:
            self._drop_slot(slot_i)
        chosen = 1 if generation is None else int(generation)
        row = DramPipelineEntry(
            req_id=key,
            slot=slot_i,
            lookup_hit_tokens=tokens,
            actual_tokens=tokens,
            prompt_tokens=tokens,
            status=STATUS_PRELOADED,
            order=self._next_order,
            generation=chosen,
        )
        self._next_order += 1
        self.items[slot_i] = row
        self._slot_of[key] = slot_i
        self._order.append(slot_i)
        self.preloaded += 1

    def remove(self, req_id: str) -> None:
        key = str(req_id)
        entry = self._find(key)
        if entry is not None:
            self._drop_slot(entry.slot)
        self._closed.add(key)

    def _drop_slot(self, slot: int) -> None:
        entry = self.items[int(slot)] if 0 <= int(slot) < _SLOTS else None
        self.items[int(slot)] = None
        self._order = [item for item in self._order if item != int(slot)]
        self._wait = [item for item in self._wait if item != int(slot)]
        if entry is not None and self._slot_of.get(entry.req_id) == int(slot):
            self._slot_of.pop(entry.req_id, None)

    def get(self, req_id: str):
        entry = self._find(req_id)
        if entry is None:
            return None
        return self._copy(entry)

    def _copy(self, entry: DramPipelineEntry) -> DramPipelineEntry:
        return DramPipelineEntry(
            req_id=entry.req_id,
            slot=entry.slot,
            lookup_hit_tokens=entry.lookup_hit_tokens,
            actual_tokens=entry.actual_tokens,
            prompt_tokens=entry.prompt_tokens,
            status=entry.status,
            order=entry.order,
            generation=entry.generation,
            enqueued_mono=entry.enqueued_mono,
            lookup_input=entry.lookup_input,
            preload_job=entry.preload_job,
            lookup_claimed=entry.lookup_claimed,
            preload_claimed=entry.preload_claimed,
            lookup_retries=entry.lookup_retries,
            retry_after=entry.retry_after,
            ticket=entry.ticket,
            reserved=entry.reserved,
        )
