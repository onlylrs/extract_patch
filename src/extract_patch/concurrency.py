from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class InflightBudget:
    """Bound both queued item count and estimated uncompressed bytes."""

    def __init__(self, max_items: int, max_bytes: int) -> None:
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("Inflight limits must be positive")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items = 0
        self._bytes = 0
        self.peak_items = 0
        self.peak_bytes = 0
        self._condition = threading.Condition()

    @contextmanager
    def reserve(self, size: int) -> Iterator[None]:
        size = max(0, min(int(size), self.max_bytes))
        with self._condition:
            self._condition.wait_for(
                lambda: self._items < self.max_items
                and (self._items == 0 or self._bytes + size <= self.max_bytes)
            )
            self._items += 1
            self._bytes += size
            self.peak_items = max(self.peak_items, self._items)
            self.peak_bytes = max(self.peak_bytes, self._bytes)
        try:
            yield
        finally:
            with self._condition:
                self._items -= 1
                self._bytes -= size
                self._condition.notify_all()
