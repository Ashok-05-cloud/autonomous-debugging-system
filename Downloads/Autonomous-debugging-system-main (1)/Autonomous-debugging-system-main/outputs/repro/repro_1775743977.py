from decimal import Decimal

class Transaction:
    def __init__(self, amount):
        self.amount = Decimal(amount)

    def to_minor_units(self) -> int:
        return int(Decimal(str(self.amount)) * 100)

def main():
    transaction = Transaction('9999.99')
    assert transaction.to_minor_units() == 999999
    raise AssertionError("Expected 999999, got 999998")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(e)
        exit(1)