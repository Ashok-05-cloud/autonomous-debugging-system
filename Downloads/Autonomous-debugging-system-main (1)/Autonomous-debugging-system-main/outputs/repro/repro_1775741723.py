from decimal import Decimal

class Transaction:
    def __init__(self, amount):
        self.amount = Decimal(amount)

    def to_minor_units(self) -> int:
        multiplier = 100
        return int(float(self.amount * multiplier))

def test_to_minor_units():
    transaction = Transaction('9999.99')
    assert transaction.to_minor_units() == 999999

    transaction = Transaction('1234.57')
    assert transaction.to_minor_units() == 123456

test_to_minor_units()