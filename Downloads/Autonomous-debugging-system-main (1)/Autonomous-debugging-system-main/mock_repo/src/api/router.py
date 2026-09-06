"""
API Layer - Request handling, rate limiting, middleware
"""
import functools
import hashlib
import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[Dict] = None
    params: Optional[Dict] = None
    client_ip: str = "127.0.0.1"
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class Response:
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter per API key / IP.

    BUG: The `_buckets` dict grows without bound because expired/inactive
    entries are never evicted. Under sustained traffic from many unique
    IPs (e.g., during a DDoS or high-traffic event), this causes:
      - Unbounded memory growth
      - Degraded performance (linear scan of large dict)
      - Eventually OOM in long-running production services

    The cleanup thread exists but has a critical bug: it acquires
    `self._lock` then calls `_cleanup_expired()` which also tries
    to acquire `self._lock` → DEADLOCK on cleanup.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: float = 300.0,  # 5 minutes
    ):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval

        # ❌ BUG: Never evicted, grows without bound
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._access_times: Dict[str, float] = {}  # track last access

        # Start cleanup thread (but has deadlock bug)
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

    def _get_or_create_bucket(self, key: str) -> Dict:
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self.burst_size,
                "last_refill": time.time(),
            }
        return self._buckets[key]

    def _refill_bucket(self, bucket: Dict) -> None:
        now = time.time()
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * self.rate
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now

    def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._refill_bucket(bucket)
            # Track access time
            self._access_times[key] = time.time()  # also never cleaned up

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, {
                    "allowed": True,
                    "remaining": int(bucket["tokens"]),
                    "reset_in": (1 - bucket["tokens"]) / self.rate,
                }
            else:
                retry_after = (1 - bucket["tokens"]) / self.rate
                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "retry_after": retry_after,
                }

    def _cleanup_expired(self, max_idle_seconds: float = 3600.0) -> int:
        """Remove buckets not accessed in max_idle_seconds."""
        # ❌ BUG: This acquires _lock but cleanup_loop ALREADY holds _lock
        # → DEADLOCK when cleanup_loop calls this method
        with self._lock:
            now = time.time()
            expired = [
                k for k, t in self._access_times.items()
                if now - t > max_idle_seconds
            ]
            for k in expired:
                del self._buckets[k]
                del self._access_times[k]
            return len(expired)

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(self.cleanup_interval)
            try:
                # ❌ BUG: Acquires lock, then calls _cleanup_expired which
                # tries to acquire SAME lock → deadlock
                with self._lock:
                    self._cleanup_expired()  # nested lock acquisition
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)


class RequestValidator:
    """Validates and sanitizes incoming requests."""

    REQUIRED_HEADERS = ["Content-Type", "X-Request-ID"]
    MAX_BODY_SIZE = 1024 * 1024  # 1MB
    ALLOWED_CONTENT_TYPES = {"application/json", "application/x-www-form-urlencoded"}

    def validate(self, ctx: RequestContext) -> Tuple[bool, Optional[str]]:
        # Check required headers (case-insensitive)
        headers_lower = {k.lower(): v for k, v in ctx.headers.items()}
        for h in self.REQUIRED_HEADERS:
            if h.lower() not in headers_lower:
                return False, f"Missing required header: {h}"

        # Validate content type for POST/PUT
        if ctx.method in ("POST", "PUT", "PATCH"):
            ct = headers_lower.get("content-type", "")
            ct_base = ct.split(";")[0].strip()
            if ct_base not in self.ALLOWED_CONTENT_TYPES:
                return False, f"Unsupported Content-Type: {ct_base}"

        return True, None


class Router:
    """Simple path-based router with middleware support."""

    def __init__(self):
        self._routes: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self._middleware: List[Callable] = []
        self.rate_limiter = RateLimiter()
        self.validator = RequestValidator()

    def route(self, method: str, path: str):
        """Decorator to register a route handler."""
        def decorator(func: Callable):
            self._routes[path][method.upper()] = func
            return func
        return decorator

    def use(self, middleware: Callable):
        self._middleware.append(middleware)

    def handle(self, ctx: RequestContext) -> Response:
        start = time.time()

        # Rate limiting
        rate_key = ctx.headers.get("X-API-Key", ctx.client_ip)
        allowed, rate_info = self.rate_limiter.is_allowed(rate_key)
        if not allowed:
            return Response(
                status_code=429,
                body={"error": "Rate limit exceeded", **rate_info},
                headers={"Retry-After": str(int(rate_info.get("retry_after", 1)))},
            )

        # Validation
        valid, err = self.validator.validate(ctx)
        if not valid:
            return Response(status_code=400, body={"error": err})

        # Route resolution
        handler_map = self._routes.get(ctx.path)
        if handler_map is None:
            return Response(status_code=404, body={"error": f"Path not found: {ctx.path}"})

        handler = handler_map.get(ctx.method.upper())
        if handler is None:
            allowed_methods = list(handler_map.keys())
            return Response(
                status_code=405,
                body={"error": "Method not allowed"},
                headers={"Allow": ", ".join(allowed_methods)},
            )

        # Run middleware chain
        for mw in self._middleware:
            result = mw(ctx)
            if isinstance(result, Response):
                return result

        # Execute handler
        try:
            response = handler(ctx)
            response.latency_ms = (time.time() - start) * 1000
            return response
        except Exception as e:
            logger.exception(f"Handler error: {ctx.method} {ctx.path}")
            return Response(
                status_code=500,
                body={"error": "Internal server error", "request_id": ctx.request_id},
            )


def require_auth(auth_service):
    """Middleware factory: enforces JWT auth."""
    def middleware(ctx: RequestContext) -> Optional[Response]:
        auth_header = ctx.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(status_code=401, body={"error": "Missing or invalid Authorization header"})
        token = auth_header[7:]
        try:
            claims = auth_service.verify_token(token)
            ctx.headers["_claims"] = claims
            return None  # continue
        except Exception as e:
            return Response(status_code=401, body={"error": str(e)})
    return middleware
