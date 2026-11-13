"""Two DRAM pools: preload and store.

Slot size is one full KVB. The caller supplies one private anonymous
mapping. Slots are cut from ``capacity``, which may be smaller than the
mapping. A span is a contiguous run of those slots. The free-list split
and coalesce follow the L1 address manager: a free interval is
``(start, size)``, and adjacent frees merge. Occupancy is ``pin_count``
plus ``preloaded`` or ``stored``. There is no ref_count.

``try_allocate`` never evicts. storeGC is the only caller of
``reclaim_to_watermark``, and it only frees store slots with ``stored``
and ``pin_count == 0``. A preload slot returns to its pool when its pin
hits zero, which is what wakes the preload thread.
"""

from __future__ import annotations

import threading


class Slot:
    def __init__(self, index: int, address: int, nbytes: int, base_ptr: int):
        self.index = int(index)
        self.address = int(address)
        self.nbytes = int(nbytes)
        self.base_ptr = int(base_ptr)
        self.buf = self.base_ptr + self.address
        self.pin_count = 0
        self.stored = False
        self.preloaded = False
        self.purpose = "free"
        self.generation = 0
        self.req_id = ""
        self.freed = True

    def pin_view(self):
        return self.buf


class StorePool:
    def __init__(
        self,
        slot_bytes: int,
        slot_count: int | None = None,
        capacity_bytes: int | None = None,
        kind: str = "store",
        on_free=None,
        watermark_bytes: int = 0,
        base_ptr: int = 0,
        mapped_bytes: int = 0,
        release=None,
    ):
        if int(slot_bytes) <= 0:
            raise ValueError("slot_bytes must be positive")
        if kind not in ("store", "preload"):
            raise ValueError("pool kind must be store or preload")
        if capacity_bytes is None:
            if slot_count is None or int(slot_count) <= 0:
                raise ValueError("slot_count or capacity_bytes is required")
            capacity_bytes = int(slot_bytes) * int(slot_count)
        capacity = int(capacity_bytes)
        if capacity < int(slot_bytes):
            raise ValueError("pool capacity is smaller than one slot")
        self.kind = kind
        self.slot_bytes = int(slot_bytes)
        self.slot_count = capacity // self.slot_bytes
        self.capacity = self.slot_count * self.slot_bytes
        self.store_cap = self.capacity
        if int(base_ptr) == 0 or int(mapped_bytes) <= 0:
            raise ValueError("host mapping is required")
        if int(mapped_bytes) < self.capacity:
            raise ValueError("mapped_bytes is smaller than pool capacity")
        if not callable(release):
            raise ValueError("release is required")
        self.watermark_bytes = max(0, int(watermark_bytes))
        self._on_free = on_free
        self._base = int(base_ptr)
        self._mapped = int(mapped_bytes)
        self._release = release
        self.host_pinned = False
        self._host_unpin = None
        self._slots = [
            Slot(i, i * self.slot_bytes, self.slot_bytes, self._base)
            for i in range(self.slot_count)
        ]
        # One free interval covering the pool. Freed neighbors coalesce.
        self._free = [(0, self.slot_count)]
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._freed = threading.Event()
        self._target = self.watermark_bytes
        self._max_block = 0
        self.reclaim_objs = 0
        self.reclaim_bytes = 0
        self.watermark_deficit = 0

    def host_memory(self):
        if not self._base:
            return None
        return int(self._base), int(self._mapped)

    def note_pinned(self, unpin) -> None:
        self.host_pinned = True
        if callable(unpin):
            self._host_unpin = unpin

    def close(self) -> None:
        unpin = self._host_unpin
        base = int(self._base or 0)
        mapped = int(self._mapped or 0)
        release = self._release
        registered = self.host_pinned
        self._host_unpin = None
        self.host_pinned = False
        self._release = None
        self._base = 0
        self._mapped = 0
        if registered and callable(unpin) and base:
            try:
                unpin(base)
            except Exception:
                pass
        if callable(release) and base:
            try:
                release(base, mapped)
            except Exception:
                pass

    def set_on_free(self, fn) -> None:
        self._on_free = fn

    def try_allocate(self, nbytes: int, purpose: str, req_id: str = "", generation: int = 0):
        """Take a contiguous free run. Does not evict."""
        need = max(0, int(nbytes))
        if need <= 0 or purpose not in ("store", "preload"):
            return None
        span = (need + self.slot_bytes - 1) // self.slot_bytes
        with self._lock:
            start = self._take_locked(span)
            if start is None:
                return None
            run = []
            for index in range(start, start + span):
                slot = self._slots[index]
                slot.purpose = purpose
                slot.stored = False
                slot.preloaded = False
                slot.pin_count = 1
                slot.freed = False
                slot.req_id = str(req_id)
                slot.generation = int(generation)
                run.append(slot)
            return run

    def pin(self, slots) -> None:
        with self._lock:
            for slot in slots:
                if not slot.freed:
                    slot.pin_count += 1

    def unpin(self, slots) -> None:
        """Drop one pin. A preload slot returns to the pool at zero."""
        notify = False
        with self._lock:
            for slot in slots:
                if slot.freed:
                    continue
                slot.pin_count = max(0, slot.pin_count - 1)
                if slot.pin_count == 0 and slot.purpose == "preload":
                    self._free_locked(slot)
                    notify = True
        if notify:
            self._note_free()

    def mark_stored(self, slots) -> None:
        with self._lock:
            for slot in slots:
                if not slot.freed and slot.purpose == "store":
                    slot.stored = True

    def mark_preloaded(self, slots) -> None:
        with self._lock:
            for slot in slots:
                if not slot.freed and slot.purpose == "preload":
                    slot.preloaded = True

    def abort_unstored(self, slots) -> None:
        """Release a cancelled span that has not been stored. Not a victim scan."""
        notify = False
        with self._lock:
            for slot in slots:
                if slot.freed or slot.stored:
                    continue
                slot.pin_count = 0
                slot.preloaded = False
                self._free_locked(slot)
                notify = True
        if notify:
            self._note_free()

    def request_watermark(self, need: int, max_block: int = 0) -> None:
        """Publish the next step's bytes and wake storeGC. Do not reclaim here."""
        need_b = max(0, int(need))
        block = max(0, int(max_block))
        with self._lock:
            self._target = max(self.watermark_bytes, need_b)
            self._max_block = block
        self._wake.set()

    def published_watermark(self):
        with self._lock:
            return int(self._target), int(self._max_block)

    def watermark_event(self) -> threading.Event:
        return self._wake

    def free_event(self) -> threading.Event:
        return self._freed

    def reclaim_to_watermark(self) -> int:
        """Free stored, unpinned store slots until the published watermark fits."""
        reclaimed = 0
        reclaimed_bytes = 0
        while not self._watermark_met():
            victim = self._pop_victim()
            if victim is None:
                with self._lock:
                    self.watermark_deficit += 1
                break
            reclaimed += 1
            reclaimed_bytes += victim.nbytes
        with self._lock:
            self.reclaim_objs += reclaimed
            self.reclaim_bytes += reclaimed_bytes
        if reclaimed:
            self._wake.set()
            self._note_free()
        return reclaimed_bytes

    def ledger_free(self) -> int:
        with self._lock:
            return self._free_bytes_locked()

    def largest_free(self) -> int:
        with self._lock:
            return self._largest_locked()

    def occupied(self, purpose: str | None = None) -> int:
        with self._lock:
            n = 0
            for slot in self._slots:
                if slot.freed:
                    continue
                if purpose is None or slot.purpose == purpose:
                    n += 1
            return n

    def _note_free(self) -> None:
        self._freed.set()
        fn = self._on_free
        if fn is not None:
            fn()

    def _watermark_met(self) -> bool:
        with self._lock:
            if self._free_bytes_locked() < int(self._target):
                return False
            if int(self._max_block) > 0 and self._largest_locked() < int(self._max_block):
                return False
            return True

    def _pop_victim(self):
        with self._lock:
            for slot in self._slots:
                if slot.freed or slot.purpose != "store":
                    continue
                if not slot.stored or slot.pin_count != 0:
                    continue
                self._free_locked(slot)
                return slot
            return None

    def _take_locked(self, span: int):
        if span <= 0 or span > self.slot_count:
            return None
        for index, (start, count) in enumerate(self._free):
            if count < span:
                continue
            self._free.pop(index)
            left = count - span
            if left:
                self._free.insert(index, (start + span, left))
            return start
        return None

    def _free_locked(self, slot: Slot) -> None:
        if slot.freed:
            return
        slot.purpose = "free"
        slot.stored = False
        slot.preloaded = False
        slot.pin_count = 0
        slot.freed = True
        slot.req_id = ""
        slot.generation = 0
        self._free.append((slot.index, 1))
        self._free.sort()
        merged = []
        for start, count in self._free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_start, prev_count = merged[-1]
                merged[-1] = (prev_start, prev_count + count)
            else:
                merged.append((start, count))
        self._free = merged

    def _free_bytes_locked(self) -> int:
        return sum(count for _, count in self._free) * self.slot_bytes

    def _largest_locked(self) -> int:
        if not self._free:
            return 0
        return max(count for _, count in self._free) * self.slot_bytes
