"""
Existing test suite for the payment processor.
These tests pass in normal scenarios but fail under edge cases.
"""
import pytest
from decimal import Decimal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.payments.processor import PaymentProcessor, PaymentStatus


@pytest.fixture
def processor():
    return PaymentProcessor(
        merchant_id="MERCHANT_001",
        secret_key="test_secret_key_do_not_use_in_prod",
        gateway_url="https://gateway.example.com/v1/charge",
    )


class TestBasicPayments:
    def test_successful_payment(self, processor):
        result = processor.process_payment(
            amount="10.00",
            currency="USD",
            customer_id="cust_001",
        )
        assert result["success"] is True
        assert result["amount"] == "10.00"
        assert result["currency"] == "USD"
        assert result["status"] == "completed"

    def test_invalid_currency(self, processor):
        result = processor.process_payment(
            amount="10.00",
            currency="XYZ",
            customer_id="cust_001",
        )
        assert result["success"] is False
        assert result["error_code"] == "INVALID_CURRENCY"

    def test_negative_amount(self, processor):
        result = processor.process_payment(
            amount="-5.00",
            currency="USD",
            customer_id="cust_001",
        )
        assert result["success"] is False

    def test_zero_amount(self, processor):
        result = processor.process_payment(
            amount="0.00",
            currency="USD",
            customer_id="cust_001",
        )
        assert result["success"] is False

    def test_refund_full(self, processor):
        pay = processor.process_payment("50.00", "USD", "cust_001")
        assert pay["success"]
        refund = processor.refund(pay["transaction_id"])
        assert refund["success"]
        assert refund["status"] == "refunded"

    def test_partial_refund(self, processor):
        pay = processor.process_payment("100.00", "USD", "cust_002")
        assert pay["success"]
        refund = processor.refund(pay["transaction_id"], Decimal("30.00"))
        assert refund["success"]
        assert refund["status"] == "partially_refunded"
        assert refund["remaining_refundable"] == "70.00"


class TestEdgeCases:
    def test_large_amount_precision(self, processor):
        """
        This test FAILS due to the float precision bug in to_minor_units().
        Decimal("9999.99") * 100 via float gives 999998 instead of 999999.
        """
        result = processor.process_payment(
            amount="9999.99",
            currency="USD",
            customer_id="cust_003",
        )
        assert result["success"] is True
        # BUG: minor_units_sent will be 999998 instead of 999999
        assert result["minor_units_sent"] == 999999, (
            f"Expected 999999 minor units but got {result['minor_units_sent']}. "
            "Float precision loss in to_minor_units()"
        )

    def test_jpy_no_subunit(self, processor):
        result = processor.process_payment(
            amount="1500",
            currency="JPY",
            customer_id="cust_004",
        )
        assert result["success"] is True
        assert result["minor_units_sent"] == 1500

    def test_rate_limit(self, processor):
        """Hit rate limit after 10 transactions."""
        for i in range(10):
            processor.process_payment("1.00", "USD", "cust_flood")
        result = processor.process_payment("1.00", "USD", "cust_flood")
        assert result["error_code"] == "RATE_LIMIT_EXCEEDED"
