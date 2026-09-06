from decimal import Decimal

class Transaction:
    def __init__(self, amount):
        self.amount = Decimal(amount)

    def to_minor_units(self) -> int:
        multiplier = 100
        return int(float(self.amount * multiplier))

def main():
    transaction = Transaction('9999.99')
    assert transaction.to_minor_units() == 999999

    transaction = Transaction('1234.57')
    assert transaction.to_minor_units() == 123456

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(e)
        exit(1)