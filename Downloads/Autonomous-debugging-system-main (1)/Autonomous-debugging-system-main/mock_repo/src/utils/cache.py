"""
Shared Utilities - Cache, retry, serialization helpers
"""
import functools
import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class TTLCache:
    """
    Thread-safe in-memory cache with per-key TTL.

    BUG: `get()` method checks expiry AFTER returning the value
    in some code paths — specifically the default_factory path
    does not check expiry before returning a freshly-computed value,
    but the bigger bug is that `_store[key]` is accessed without
    holding the lock in the fast-path read, creating a TOCTOU:

      Thread A: `key in self._store` → True
      Thread B: deletes key during eviction
      Thread A: `self._store[key]` → KeyError (unhandled)
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._store: Dict[str, Tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: Any = None, default_factory: Optional[Callable] = None) -> Any:
        """
        Get value from cache.

        BUG: The check `key in self._store` and `self._store[key]`
        are not atomic — another thread can evict the key between them.
        """
        # ❌ BUG: No lock held here
        if key in self._store:  # check
            value, expires_at = self._store[key]  # ← KeyError possible here
            if time.time() < expires_at:
                self._hits += 1
                return value
            else:
                # Expired — remove
                with self._lock:
                    self._store.pop(key, None)

        self._misses += 1

        if default_factory is not None:
            value = default_factory()
            self.set(key, value)  # Cache the result
            return value

        return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            # Evict oldest if at max capacity
            if len(self._store) >= self.max_size and key not in self._store:
                # Simple eviction: remove first expired, or oldest
                now = time.time()
                expired = [k for k, (_, exp) in self._store.items() if exp < now]
                if expired:
                    del self._store[expired[0]]
                else:
                    # Evict arbitrary entry (not LRU — another bug)
                    oldest_key = next(iter(self._store))
                    del self._store[oldest_key]

            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> Dict:
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {last_exc}")
            raise last_exc
        return wrapper
    return decorator


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def generate_correlation_id() -> str:
    """Generate a unique correlation/request ID."""
    import os
    return hashlib.sha256(os.urandom(32)).hexdigest()[:16]


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts, override wins on conflict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
