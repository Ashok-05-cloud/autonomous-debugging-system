"""
Connection Pool Manager
Manages a pool of database connections with connection lifecycle management.
"""
import threading
import time
import logging
from typing import Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    id: int
    host: str
    port: int
    database: str
    in_use: bool = False
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    _conn: object = None  # underlying DB connection

    def is_stale(self, max_age: float = 300.0) -> bool:
        return (time.time() - self.last_used) > max_age

    def execute(self, query: str, params=None):
        """Execute a query on this connection."""
        if self._conn is None:
            raise RuntimeError(f"Connection {self.id} has no underlying DB connection")
        # Simulate DB execution
        self.last_used = time.time()
        self.use_count += 1
        return MockCursor(query, params)


class MockCursor:
    def __init__(self, query, params):
        self.query = query
        self.params = params
        self._results = []

    def fetchall(self):
        return self._results

    def fetchone(self):
        return self._results[0] if self._results else None


class ConnectionPool:
    """
    Thread-safe connection pool with configurable min/max connections.

    BUG: Race condition in acquire() — check-then-act on `in_use` flag
    is not atomic. Under high concurrency, two threads can both see
    in_use=False and claim the same connection, leading to:
      - Shared connection state corruption
      - Double-release assertions
      - Silent query result mixing
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 5.0,
        max_connection_age: float = 300.0,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_connection_age = max_connection_age

        self._pool: List[Connection] = []
        self._lock = threading.Lock()  # exists but NOT used in acquire() — THE BUG
        self._conn_id_counter = 0
        self._waiters = 0
        self._total_acquired = 0
        self._total_released = 0

        # Initialize minimum connections
        for _ in range(self.min_connections):
            self._pool.append(self._create_connection())

    def _create_connection(self) -> Connection:
        self._conn_id_counter += 1
        conn = Connection(
            id=self._conn_id_counter,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        # Simulate actual DB connect
        conn._conn = object()
        logger.debug(f"Created connection {conn.id}")
        return conn

    def acquire(self) -> Optional[Connection]:
        """
        Acquire a connection from the pool.

        BUG IS HERE: The loop body reads in_use and sets it WITHOUT holding _lock.
        Between reading `not conn.in_use` and setting `conn.in_use = True`,
        another thread can see the same conn as free and take it too.
        """
        deadline = time.time() + self.connection_timeout
        self._waiters += 1

        try:
            while time.time() < deadline:
                # ❌ BUG: No lock held during this critical section
                for conn in self._pool:
                    if not conn.in_use:  # read
                        conn.in_use = True  # write — NOT atomic with read above
                        self._total_acquired += 1
                        logger.debug(f"Acquired connection {conn.id}")
                        return conn

                # Try to grow the pool
                with self._lock:
                    if len(self._pool) < self.max_connections:
                        new_conn = self._create_connection()
                        new_conn.in_use = True
                        self._pool.append(new_conn)
                        self._total_acquired += 1
                        return new_conn

                time.sleep(0.01)

        finally:
            self._waiters -= 1

        logger.warning("Connection pool exhausted, timeout reached")
        return None

    def release(self, conn: Connection) -> None:
        """Release a connection back to the pool."""
        if conn is None:
            return

        # Stale connection handling
        if conn.is_stale(self.max_connection_age):
            logger.info(f"Retiring stale connection {conn.id}")
            self._retire_connection(conn)
            # Ensure we maintain minimum pool size
            with self._lock:
                if len(self._pool) < self.min_connections:
                    self._pool.append(self._create_connection())
            return

        conn.in_use = False
        conn.last_used = time.time()
        self._total_released += 1
        logger.debug(f"Released connection {conn.id}")

    def _retire_connection(self, conn: Connection) -> None:
        with self._lock:
            if conn in self._pool:
                self._pool.remove(conn)
                conn._conn = None
                logger.debug(f"Retired connection {conn.id}")

    @contextmanager
    def get_connection(self):
        """Context manager for safe connection use."""
        conn = self.acquire()
        if conn is None:
            raise TimeoutError(
                f"Could not acquire connection within {self.connection_timeout}s. "
                f"Pool size: {len(self._pool)}, waiters: {self._waiters}"
            )
        try:
            yield conn
        except Exception as exc:
            logger.error(f"Error using connection {conn.id}: {exc}")
            raise
        finally:
            self.release(conn)

    def stats(self) -> dict:
        in_use = sum(1 for c in self._pool if c.in_use)
        return {
            "total_connections": len(self._pool),
            "in_use": in_use,
            "available": len(self._pool) - in_use,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "waiters": self._waiters,
        }

    def close_all(self) -> None:
        with self._lock:
            for conn in self._pool:
                conn._conn = None
            self._pool.clear()
        logger.info("All connections closed")
