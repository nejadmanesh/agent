"""Utilities for tracking the memory usage of loaded models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Iterator


@dataclass(slots=True)
class MemoryReservation:
    """Represents a reservation of model memory that must be released."""

    size_bytes: int
    owner: str


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Information about current memory usage in the inference runtime."""

    limit_bytes: int
    used_bytes: int

    @property
    def available_bytes(self) -> int:
        return max(self.limit_bytes - self.used_bytes, 0)

    @property
    def usage_ratio(self) -> float:
        if self.limit_bytes == 0:
            return 0.0
        return self.used_bytes / self.limit_bytes


class MemoryManager:
    """Thread-safe manager to keep track of memory reservations."""

    def __init__(self, limit_bytes: int) -> None:
        if limit_bytes < 0:
            raise ValueError("The memory limit must be non-negative.")
        self._limit_bytes = limit_bytes
        self._used_bytes = 0
        self._lock = Lock()

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def status(self) -> MemoryStatus:
        return MemoryStatus(limit_bytes=self._limit_bytes, used_bytes=self._used_bytes)

    def reserve(self, size_bytes: int, *, owner: str) -> MemoryReservation:
        if size_bytes < 0:
            raise ValueError("Reservation size must be non-negative.")
        with self._lock:
            if self._used_bytes + size_bytes > self._limit_bytes:
                raise MemoryError(
                    f"Unable to reserve {size_bytes} bytes for {owner}; "
                    f"limit of {self._limit_bytes} bytes would be exceeded."
                )
            self._used_bytes += size_bytes
        return MemoryReservation(size_bytes=size_bytes, owner=owner)

    def release(self, reservation: MemoryReservation) -> None:
        with self._lock:
            self._used_bytes = max(self._used_bytes - reservation.size_bytes, 0)

    @contextmanager
    def scoped_reservation(self, size_bytes: int, *, owner: str) -> Iterator[MemoryReservation]:
        reservation = self.reserve(size_bytes, owner=owner)
        try:
            yield reservation
        finally:
            self.release(reservation)
