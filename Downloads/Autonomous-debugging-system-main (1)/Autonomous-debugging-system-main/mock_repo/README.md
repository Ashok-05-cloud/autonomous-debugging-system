# FinTech Platform - Mock Repository

A realistic multi-module fintech platform with intentionally introduced bugs for AI debugger testing.

## Architecture

```
src/
├── auth/          # JWT authentication & RBAC
├── payments/      # Payment processing engine
├── database/      # Connection pool management
├── api/           # HTTP routing & rate limiting
└── utils/         # Shared utilities (cache, retry)
```

## Known Bugs (for testing)

| Bug ID | Location | Type | Severity |
|--------|----------|------|----------|
| BUG-2847 | `src/payments/processor.py:58` | Float precision in `to_minor_units()` | CRITICAL |
| BUG-2801 | `src/database/connection_pool.py:89` | Race condition in `acquire()` | HIGH |
| BUG-2756 | `src/auth/auth_service.py` | Negative JWT leeway causing premature expiry | HIGH |
| BUG-2799 | `src/api/router.py` | Memory leak + deadlock in RateLimiter cleanup | MEDIUM |
| BUG-2812 | `src/utils/cache.py` | TOCTOU race in TTLCache.get() | MEDIUM |

## Running Tests

```bash
cd mock_repo
pip install pytest
pytest tests/ -v
```

## Bug Report & Logs

- `bug_report.json` — Structured bug report for BUG-2847
- `logs/production.log` — Production logs with stack traces and noise
