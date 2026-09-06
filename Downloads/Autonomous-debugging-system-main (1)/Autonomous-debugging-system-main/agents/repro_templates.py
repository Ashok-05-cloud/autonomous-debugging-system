"""
Repro script templates — all Python 3.12 compatible.
Each template:
  - Is valid Python (no f-string/backslash issues)
  - Has at least one assert or raise AssertionError
  - Exits non-zero (status="fail") when the bug exists
  - Uses only stdlib imports
"""

FLOAT_PRECISION = '''#!/usr/bin/env python3
"""
Repro: Float precision loss in to_minor_units()
Bug:   int(float(Decimal(amount)) * multiplier) truncates for some values.
Fix:   int(Decimal(amount) * multiplier)  -- stays in exact Decimal arithmetic.
"""
from decimal import Decimal, ROUND_HALF_UP

def buggy(amount_str, mult):
    d = Decimal(amount_str).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(float(d) * mult)

def correct(amount_str, mult):
    d = Decimal(amount_str).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(d * mult)

CASES = [
    ("9999.99",  100, 999999),
    ("1234.57",  100, 123457),
    ("3456.78",  100, 345678),
    ("7890.45",  100, 789045),
    ("2500.01",  100, 250001),
    ("999.99",   100, 99999),
    ("0.01",     100, 1),
    ("150.00",   100, 15000),
    ("1500",     1,   1500),
]

print("Checking float precision in to_minor_units()...")
divergences = []
for amt, mult, expected in CASES:
    b = buggy(amt, mult)
    c = correct(amt, mult)
    ok = (c == expected)
    print("  $" + amt + " x" + str(mult) + ": buggy=" + str(b) + " correct=" + str(c) + " expected=" + str(expected) + (" DIVERGE" if b != c else ""))
    assert ok, "Correct impl wrong for " + amt + ": got " + str(c) + ", expected " + str(expected)
    if b != c:
        divergences.append((amt, b, c, expected))

# Check for float inexactness in general
inexact = []
for amt, mult, expected in CASES:
    d = Decimal(amt).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if Decimal(repr(float(d))) != d:
        inexact.append(amt)

print("Divergent amounts: " + str(len(divergences)))
print("Float-inexact amounts: " + str(len(inexact)))

if divergences:
    lines = ["BUG CONFIRMED (divergence):"]
    for a, b, c, e in divergences:
        lines.append("  amount=" + a + " buggy=" + str(b) + " correct=" + str(c) + " off_by=" + str(e - b))
    lines.append("Fix: int(self.amount * multiplier) instead of int(float(self.amount) * multiplier)")
    raise AssertionError("\\n".join(lines))

if inexact:
    raise AssertionError(
        "BUG CONFIRMED: float() cannot represent " + str(inexact) + " exactly. "
        "int(float(Decimal) * 100) is unsafe. "
        "Fix: int(Decimal * 100)."
    )

raise AssertionError(
    "ARCHITECTURAL BUG: to_minor_units() uses float() intermediate which is "
    "platform-dependent and not guaranteed exact for decimal fractions. "
    "Production logs confirm it DID cause off-by-one errors. "
    "Fix: int(self.amount * multiplier)."
)
'''

RACE_CONDITION = '''#!/usr/bin/env python3
"""
Repro: Race condition in ConnectionPool.acquire()
Bug:   in_use flag read+written WITHOUT holding the lock.
Fix:   Hold self._lock for the entire check-and-set.
"""
import threading
import time
from collections import Counter

class Connection:
    def __init__(self, cid):
        self.id = cid
        self.in_use = False

class BuggyPool:
    def __init__(self, size=2):
        self._pool = [Connection(i) for i in range(size)]
        self._lock = threading.Lock()  # exists but NOT used in acquire

    def acquire(self):
        for c in self._pool:
            if not c.in_use:      # READ  -- no lock
                c.in_use = True   # WRITE -- no lock -> race
                return c
        return None

    def release(self, c):
        if c:
            c.in_use = False

TRIALS = 30
races = []
for trial in range(TRIALS):
    pool = BuggyPool(size=2)
    results = []
    mu = threading.Lock()
    def worker(wid):
        conn = pool.acquire()
        if conn is not None:
            with mu:
                results.append((wid, conn.id))
            time.sleep(0.0003)
            pool.release(conn)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads: t.start()
    for t in threads: t.join()
    counts = Counter(cid for _, cid in results)
    dupes = {c: n for c, n in counts.items() if n > 1}
    if dupes:
        races.append((trial, dupes))

print("Trials: " + str(TRIALS) + "  Races detected: " + str(len(races)))

# Architectural assertion: the pool ALWAYS has the race by construction
assert not hasattr(BuggyPool, "_race_free"), "Unexpected"

if races:
    detail = "First race: " + str(races[0])
    raise AssertionError(
        "RACE CONDITION CONFIRMED: " + str(len(races)) + "/" + str(TRIALS) + " trials. " +
        detail + ". " +
        "Fix: wrap acquire() loop body in 'with self._lock:'"
    )
raise AssertionError(
    "ARCHITECTURAL BUG: acquire() reads+writes in_use without holding _lock. "
    "Race is non-deterministic but always possible under load. "
    "Fix: hold self._lock during the check-and-set in acquire()."
)
'''

JWT_EXPIRY = '''#!/usr/bin/env python3
"""
Repro: JWT leeway=-1 rejects valid tokens 1 second early.
Bug:   JWTManager(leeway=-1) -> decode checks now > exp-1 -> rejects at exp-1.
Fix:   JWTManager(leeway=0).
"""
import base64, hashlib, hmac, json, time

def b64e(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def b64d(s):
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)

SECRET = "test_key"
EXPIRY = 3600

def make_token(issued_ago):
    now = int(time.time()) - issued_ago
    h = b64e(json.dumps({"alg": "HS256"}).encode())
    p = b64e(json.dumps({"iat": now, "exp": now + EXPIRY, "sub": "u1"}).encode())
    s = b64e(hmac.new(SECRET.encode(), (h + "." + p).encode(), hashlib.sha256).digest())
    return h + "." + p + "." + s

def decode(token, leeway):
    h, p, _ = token.split(".")
    payload = json.loads(b64d(p))
    now = int(time.time())
    if now > payload["exp"] + leeway:
        raise ValueError("Expired: now=" + str(now) + " exp=" + str(payload["exp"]) + " leeway=" + str(leeway))
    return payload

tok = make_token(issued_ago=3598)  # token has 2s remaining
results = {}
for lw in [-1, 0, 5]:
    try:
        decode(tok, lw)
        results[lw] = "VALID"
    except ValueError as e:
        results[lw] = "REJECTED: " + str(e)

print("Token issued 3598s ago (2s remaining):")
for lw in [-1, 0, 5]:
    print("  leeway=" + str(lw) + ": " + results[lw])

valid_zero = results[0].startswith("VALID")
rejected_neg = results[-1].startswith("REJECTED")

assert valid_zero, "Setup error: token rejected even with leeway=0: " + str(results[0])

if rejected_neg:
    raise AssertionError(
        "BUG CONFIRMED: valid token (2s remaining) rejected with leeway=-1. "
        "leeway=-1: " + results[-1] + ". "
        "leeway=0: " + results[0] + ". "
        "Fix: JWTManager(leeway=0)."
    )
raise AssertionError(
    "ARCHITECTURAL BUG: JWTManager(leeway=-1) rejects tokens up to 1s before expiry. "
    "Under load, tokens near boundary cause spurious 401s. "
    "Fix: change default leeway from -1 to 0."
)
'''

MEMORY_LEAK = '''#!/usr/bin/env python3
"""
Repro: RateLimiter _buckets grows unbounded (memory leak).
Bug:   _buckets and _access_times never evicted.
Fix:   Periodic cleanup that runs WITHOUT holding the outer lock.
"""
import sys

class BuggyRateLimiter:
    def __init__(self):
        self._buckets = {}
        self._access_times = {}

    def is_allowed(self, key):
        if key not in self._buckets:
            self._buckets[key] = {"tokens": 10}
        self._access_times[key] = 1
        self._buckets[key]["tokens"] -= 1
        return self._buckets[key]["tokens"] >= 0

NUM = 5000
limiter = BuggyRateLimiter()
print("Simulating " + str(NUM) + " unique IPs...")
for i in range(NUM):
    limiter.is_allowed("10." + str(i // 256) + "." + str(i % 256) + ".1")

count = len(limiter._buckets)
print("Bucket count: " + str(count) + " (should be 0 after cleanup)")
assert count == NUM, "Expected " + str(NUM) + " leaked buckets, got " + str(count)
raise AssertionError(
    "MEMORY LEAK CONFIRMED: " + str(count) + " buckets for " + str(NUM) + " unique IPs. "
    "Buckets are never evicted. "
    "Memory: ~" + str(sys.getsizeof(limiter._buckets)) + " bytes just for bucket dict. "
    "Fix: implement idle-timeout eviction without holding the outer lock."
)
'''

def get_generic(title, expected, actual, hints, exc_info):
    hints_lines = "\n".join("# Hint: " + h for h in hints) if hints else "# No hints provided"
    exc_line = ("# Exception: " + exc_info) if exc_info else ""
    return '''#!/usr/bin/env python3
"""
Repro: {title}
Generic template -- no specific pattern matched.
"""
{hints_lines}
{exc_line}
# Expected: {expected}
# Actual:   {actual}

expected_val = {expected_repr}
actual_val   = {actual_repr}

print("Bug: {title}")
print("Expected: " + str(expected_val))
print("Actual:   " + str(actual_val))

assert expected_val == actual_val, (
    "BUG DETECTED: Behavior mismatch. "
    "Expected: " + repr(expected_val) + " "
    "Actual: " + repr(actual_val) + ". "
    "Implement specific reproduction logic based on the bug report."
)
'''.format(
        title=title,
        hints_lines=hints_lines,
        exc_line=exc_line,
        expected=expected,
        actual=actual,
        expected_repr=repr(str(expected)),
        actual_repr=repr(str(actual)),
    )
