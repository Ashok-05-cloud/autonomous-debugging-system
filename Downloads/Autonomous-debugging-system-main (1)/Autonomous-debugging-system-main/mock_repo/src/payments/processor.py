"""
Payment Processing Module
Handles transaction processing, refunds, and payment validation.
"""
import hashlib
import hmac
import time
import logging
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


CURRENCY_MULTIPLIERS = {
    Currency.USD: 100,   # cents
    Currency.EUR: 100,   # cents
    Currency.GBP: 100,   # pence
    Currency.JPY: 1,     # no subunit
}

MAX_TRANSACTION_AMOUNT = Decimal("999999.99")
MIN_TRANSACTION_AMOUNT = Decimal("0.01")


@dataclass
class Transaction:
    id: str
    amount: Decimal
    currency: Currency
    status: PaymentStatus
    merchant_id: str
    customer_id: str
    created_at: float
    metadata: Dict[str, Any] = None
    refunded_amount: Decimal = Decimal("0.00")

    def to_minor_units(self) -> int:
        """
        Convert amount to minor currency units (e.g., dollars → cents).

        BUG: For large amounts near MAX_TRANSACTION_AMOUNT, the int conversion
        after multiplying can silently overflow in environments/languages where
        this is an issue, but the real Python bug here is that we use float
        multiplication BEFORE converting to int, causing precision loss.

        E.g., Decimal("9999.99") * 100 via float = 999998 instead of 999999
        """
        multiplier = CURRENCY_MULTIPLIERS[self.currency]
        # ❌ BUG: Converting to float loses precision for large Decimal values

        return int(self.amount * multiplier)

    @property
    def refundable_amount(self) -> Decimal:
        return self.amount - self.refunded_amount


class PaymentProcessor:
    """
    Core payment processing engine.
    Handles authorization, capture, and refund workflows.
    """

    def __init__(self, merchant_id: str, secret_key: str, gateway_url: str):
        self.merchant_id = merchant_id
        self._secret_key = secret_key
        self.gateway_url = gateway_url
        self._transactions: Dict[str, Transaction] = {}
        self._rate_limits: Dict[str, list] = {}  # customer_id → [timestamps]

    def _generate_transaction_id(self) -> str:
        ts = str(time.time()).encode()
        return hashlib.sha256(ts + self.merchant_id.encode()).hexdigest()[:16].upper()

    def _sign_payload(self, payload: Dict) -> str:
        """HMAC-SHA256 signature for payload integrity."""
        message = "&".join(f"{k}={v}" for k, v in sorted(payload.items()))
        return hmac.new(
            self._secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _check_rate_limit(self, customer_id: str, window: float = 60.0, max_txn: int = 10) -> bool:
        now = time.time()
        timestamps = self._rate_limits.get(customer_id, [])
        # Purge old timestamps
        timestamps = [t for t in timestamps if now - t < window]
        if len(timestamps) >= max_txn:
            return False
        timestamps.append(now)
        self._rate_limits[customer_id] = timestamps
        return True

    def process_payment(
        self,
        amount: Any,
        currency: str,
        customer_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process a payment transaction.
        Returns structured result dict.
        """
        # Validate and normalize amount
        try:
            amount_decimal = Decimal(str(amount)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, TypeError) as e:
            return self._error_response(f"Invalid amount: {e}", "INVALID_AMOUNT")

        # Validate currency
        try:
            currency_enum = Currency(currency.upper())
        except ValueError:
            return self._error_response(f"Unsupported currency: {currency}", "INVALID_CURRENCY")

        # Amount bounds check
        if amount_decimal < MIN_TRANSACTION_AMOUNT:
            return self._error_response(
                f"Amount below minimum ({MIN_TRANSACTION_AMOUNT})", "AMOUNT_TOO_SMALL"
            )
        if amount_decimal > MAX_TRANSACTION_AMOUNT:
            return self._error_response(
                f"Amount exceeds maximum ({MAX_TRANSACTION_AMOUNT})", "AMOUNT_TOO_LARGE"
            )

        # Rate limiting
        if not self._check_rate_limit(customer_id):
            return self._error_response(
                "Rate limit exceeded for customer", "RATE_LIMIT_EXCEEDED"
            )

        txn_id = self._generate_transaction_id()
        txn = Transaction(
            id=txn_id,
            amount=amount_decimal,
            currency=currency_enum,
            status=PaymentStatus.PROCESSING,
            merchant_id=self.merchant_id,
            customer_id=customer_id,
            created_at=time.time(),
            metadata=metadata or {},
        )

        # Convert to minor units for gateway (THIS IS WHERE THE BUG MANIFESTS)
        minor_units = txn.to_minor_units()

        # Simulate gateway call
        try:
            gateway_result = self._call_gateway(txn_id, minor_units, currency_enum)
        except Exception as e:
            txn.status = PaymentStatus.FAILED
            self._transactions[txn_id] = txn
            logger.error(f"Gateway error for txn {txn_id}: {e}")
            return self._error_response(str(e), "GATEWAY_ERROR")

        txn.status = PaymentStatus.COMPLETED
        self._transactions[txn_id] = txn

        return {
            "success": True,
            "transaction_id": txn_id,
            "amount": str(amount_decimal),
            "currency": currency_enum.value,
            "status": txn.status.value,
            "minor_units_sent": minor_units,
            "timestamp": txn.created_at,
        }

    def _call_gateway(self, txn_id: str, minor_units: int, currency: Currency) -> Dict:
        """Simulate gateway API call."""
        if minor_units <= 0:
            raise ValueError(f"Invalid minor_units value: {minor_units}")
        # Simulate network latency
        return {"gateway_txn_id": f"GW_{txn_id}", "authorized": True}

    def refund(self, transaction_id: str, refund_amount: Optional[Decimal] = None) -> Dict:
        """Process a refund for a completed transaction."""
        txn = self._transactions.get(transaction_id)
        if not txn:
            return self._error_response(f"Transaction not found: {transaction_id}", "NOT_FOUND")

        if txn.status not in (PaymentStatus.COMPLETED, PaymentStatus.PARTIALLY_REFUNDED):
            return self._error_response(
                f"Cannot refund transaction in status: {txn.status.value}", "INVALID_STATUS"
            )

        refund_amount = refund_amount or txn.refundable_amount
        if refund_amount > txn.refundable_amount:
            return self._error_response(
                f"Refund amount {refund_amount} exceeds refundable {txn.refundable_amount}",
                "REFUND_EXCEEDS_ORIGINAL",
            )

        txn.refunded_amount += refund_amount
        txn.status = (
            PaymentStatus.REFUNDED
            if txn.refunded_amount >= txn.amount
            else PaymentStatus.PARTIALLY_REFUNDED
        )

        return {
            "success": True,
            "transaction_id": transaction_id,
            "refunded_amount": str(refund_amount),
            "remaining_refundable": str(txn.refundable_amount),
            "status": txn.status.value,
        }

    def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        txn = self._transactions.get(transaction_id)
        if not txn:
            return None
        return {
            "id": txn.id,
            "amount": str(txn.amount),
            "currency": txn.currency.value,
            "status": txn.status.value,
            "merchant_id": txn.merchant_id,
            "customer_id": txn.customer_id,
            "created_at": txn.created_at,
            "refunded_amount": str(txn.refunded_amount),
        }

    @staticmethod
    def _error_response(message: str, code: str) -> Dict:
        return {
            "success": False,
            "error": message,
            "error_code": code,
        }
