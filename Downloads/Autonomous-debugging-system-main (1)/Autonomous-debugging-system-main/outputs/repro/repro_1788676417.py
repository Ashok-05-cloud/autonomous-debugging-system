from decimal import Decimal

class Transaction:
    def __init__(self, amount):
        self.amount = Decimal(amount)

    def to_minor_units(self, multiplier=100):
        return int(float(self.amount) * multiplier)

def test_to_minor_units():
    transaction = Transaction('9999.99')
    assert transaction.to_minor_units() == 999999, f"Expected 999999, got {transaction.to_minor_units()}"

def test_to_minor_units_rounding():
    transaction = Transaction('1234.57')
    assert transaction.to_minor_units() == 123449, f"Expected 123449, got {transaction.to_minor_units()}"

def main():
    try:
        test_to_minor_units()
        test_to_minor_units_rounding()
    except AssertionError as e:
        print(e)
        raise

if __name__ == "__main__":
    main()