#!/usr/bin/env python3
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
    raise AssertionError("\n".join(lines))

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