from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

@dataclass
class Transaction:
    amount: Decimal
    payment_status: PaymentStatus

    def to_minor_units(self):
        return int(self.amount * 100)

def test_to_minor_units():
    transaction = Transaction(Decimal('9999.99'), PaymentStatus.COMPLETED)
    assert transaction.to_minor_units() == 999999, f"Expected 999999, got {transaction.to_minor_units()}"

def test_to_minor_units_rounding():
    transaction = Transaction(Decimal('1234.57'), PaymentStatus.COMPLETED)
    assert transaction.to_minor_units() == 123457, f"Expected 123457, got {transaction.to_minor_units()}"

def test_to_minor_units_invalid():
    transaction = Transaction(Decimal('9999.98'), PaymentStatus.COMPLETED)
    assert False, f"Expected an error, got {transaction.to_minor_units()}"

test_to_minor_units()
test_to_minor_units_rounding()
try:
    test_to_minor_units_invalid()
    assert False, "Expected an error to be raised"
except InvalidOperation:
    pass