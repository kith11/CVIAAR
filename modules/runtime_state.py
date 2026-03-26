import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: Any
    expires_at: float | None
    updated_at: float


class ThreadSafeTTLStore:
    """A small in-process TTL store with coarse locking for app runtime state."""

    def __init__(self, maxsize: int = 0, default_ttl: float | None = None):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._entries: OrderedDict[Any, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    def _now(self) -> float:
        return time.monotonic()

    def _resolve_expiry(self, ttl: float | None) -> float | None:
        effective_ttl = self.default_ttl if ttl is None else ttl
        if effective_ttl is None:
            return None
        return self._now() + effective_ttl

    def _prune_locked(self) -> None:
        now = self._now()
        expired = [
            key for key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)

        while self.maxsize > 0 and len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            self._prune_locked()
            entry = self._entries.get(key)
            if entry is None:
                return default
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: Any, value: Any, ttl: float | None = None) -> Any:
        with self._lock:
            self._entries[key] = _Entry(
                value=value,
                expires_at=self._resolve_expiry(ttl),
                updated_at=self._now(),
            )
            self._entries.move_to_end(key)
            self._prune_locked()
            return value

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            self._prune_locked()
            entry = self._entries.pop(key, None)
            if entry is None:
                return default
            return entry.value

    def remove(self, key: Any) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def increment(self, key: Any, delta: int = 1, ttl: float | None = None, initial: int = 0) -> int:
        with self._lock:
            self._prune_locked()
            entry = self._entries.get(key)
            current = initial if entry is None else entry.value
            new_value = int(current) + delta
            self._entries[key] = _Entry(
                value=new_value,
                expires_at=self._resolve_expiry(ttl),
                updated_at=self._now(),
            )
            self._entries.move_to_end(key)
            self._prune_locked()
            return new_value

    def snapshot(self) -> dict[Any, Any]:
        with self._lock:
            self._prune_locked()
            return {key: entry.value for key, entry in self._entries.items()}

    def items(self) -> list[tuple[Any, Any]]:
        return list(self.snapshot().items())

    def contains(self, key: Any) -> bool:
        with self._lock:
            self._prune_locked()
            return key in self._entries

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
