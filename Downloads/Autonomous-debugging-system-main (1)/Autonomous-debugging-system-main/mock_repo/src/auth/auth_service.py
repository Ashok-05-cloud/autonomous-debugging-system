"""
Authentication & Authorization Module
JWT-based auth with role-based access control.
"""
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# JWT configuration
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour
REFRESH_TOKEN_EXPIRY_SECONDS = 86400 * 30  # 30 days


class Role(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


ROLE_PERMISSIONS = {
    Role.ADMIN: ["read", "write", "delete", "admin", "billing"],
    Role.MANAGER: ["read", "write", "billing"],
    Role.USER: ["read", "write"],
    Role.READONLY: ["read"],
    Role.SERVICE: ["read", "write", "service_call"],
}


@dataclass
class UserRecord:
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[Role]
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    mfa_secret: Optional[str] = None


class AuthError(Exception):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


class TokenExpiredError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass


class PermissionDeniedError(AuthError):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    # Add padding
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


class JWTManager:
    """
    Lightweight JWT implementation using HMAC-SHA256.

    BUG: Token expiry check uses `issued_at + TOKEN_EXPIRY_SECONDS`
    but compares against `time.time()` with an off-by-one where
    tokens issued at exactly the boundary are treated as valid
    for one additional second. Worse: the `leeway` parameter
    defaults to -1 (negative), which means tokens expire 1 second
    BEFORE they should — causing valid tokens to be rejected,
    leading to spurious 401s in production under load.
    """

    def __init__(self, secret_key: str, leeway: int = -1):  # ❌ BUG: leeway=-1 should be 0 or positive
        self._secret = secret_key
        self.leeway = leeway  # negative leeway = tokens expire early

    def _sign(self, header_b64: str, payload_b64: str) -> str:
        message = f"{header_b64}.{payload_b64}".encode()
        sig = hmac.new(self._secret.encode(), message, hashlib.sha256).digest()
        return _b64url_encode(sig)

    def encode(self, payload: Dict[str, Any]) -> str:
        """Encode a JWT token."""
        header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
        now = int(time.time())
        full_payload = {
            "iat": now,
            "exp": now + TOKEN_EXPIRY_SECONDS,
            **payload,
        }

        header_b64 = _b64url_encode(json.dumps(header).encode())
        payload_b64 = _b64url_encode(json.dumps(full_payload).encode())
        signature = self._sign(header_b64, payload_b64)

        return f"{header_b64}.{payload_b64}.{signature}"

    def decode(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        Raises TokenExpiredError or InvalidTokenError on failure.
        """
        try:
            parts = token.strip().split(".")
            if len(parts) != 3:
                raise InvalidTokenError("Malformed token: wrong number of segments", "MALFORMED_TOKEN")

            header_b64, payload_b64, provided_sig = parts

            # Verify signature
            expected_sig = self._sign(header_b64, payload_b64)
            if not hmac.compare_digest(expected_sig, provided_sig):
                raise InvalidTokenError("Token signature verification failed", "INVALID_SIGNATURE")

            # Decode payload
            try:
                payload = json.loads(_b64url_decode(payload_b64))
            except Exception:
                raise InvalidTokenError("Cannot decode token payload", "DECODE_ERROR")

            # Expiry check — BUG: leeway=-1 causes valid tokens to fail
            now = int(time.time())
            exp = payload.get("exp", 0)
            # ❌ BUG: `+ self.leeway` with leeway=-1 means we reject tokens
            # that still have 1 second of validity left
            if now > exp + self.leeway:
                raise TokenExpiredError(
                    f"Token expired at {exp}, current time {now}",
                    "TOKEN_EXPIRED",
                )

            return payload

        except (TokenExpiredError, InvalidTokenError):
            raise
        except Exception as e:
            raise InvalidTokenError(f"Token processing error: {e}", "PROCESSING_ERROR")


class AuthService:
    """
    Full authentication service with login, token management, and RBAC.
    """

    def __init__(self, secret_key: str):
        self._jwt = JWTManager(secret_key)
        self._users: Dict[str, UserRecord] = {}
        self._refresh_tokens: Dict[str, Dict] = {}  # token → metadata
        self._session_blacklist: set = set()
        self.MAX_FAILED_ATTEMPTS = 5
        self.LOCKOUT_DURATION = 300  # 5 minutes

    def _hash_password(self, password: str, salt: Optional[str] = None) -> str:
        if salt is None:
            import os
            salt = os.urandom(16).hex()
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return f"{salt}:{dk.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            salt, _ = stored_hash.split(":", 1)
            expected = self._hash_password(password, salt)
            return hmac.compare_digest(expected, stored_hash)
        except Exception:
            return False

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str],
    ) -> Dict:
        if any(u.username == username for u in self._users.values()):
            return {"success": False, "error": "Username already exists"}
        if any(u.email == email for u in self._users.values()):
            return {"success": False, "error": "Email already registered"}

        try:
            role_enums = [Role(r.lower()) for r in roles]
        except ValueError as e:
            return {"success": False, "error": f"Invalid role: {e}"}

        user_id = hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()[:16]
        user = UserRecord(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            roles=role_enums,
        )
        self._users[user_id] = user
        logger.info(f"Registered user {username} (id={user_id})")
        return {"success": True, "user_id": user_id}

    def login(self, username: str, password: str) -> Dict:
        """Authenticate user and return access + refresh tokens."""
        user = next((u for u in self._users.values() if u.username == username), None)
        if not user:
            return {"success": False, "error": "Invalid credentials", "code": "INVALID_CREDENTIALS"}

        # Check lockout
        if user.locked_until and time.time() < user.locked_until:
            remaining = int(user.locked_until - time.time())
            return {
                "success": False,
                "error": f"Account locked for {remaining}s",
                "code": "ACCOUNT_LOCKED",
            }

        if not user.is_active:
            return {"success": False, "error": "Account disabled", "code": "ACCOUNT_DISABLED"}

        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1
            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = time.time() + self.LOCKOUT_DURATION
                logger.warning(f"Account locked: {username} after {user.failed_attempts} failed attempts")
            return {"success": False, "error": "Invalid credentials", "code": "INVALID_CREDENTIALS"}

        # Success — reset failed attempts
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = time.time()

        access_token = self._jwt.encode({
            "sub": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "type": "access",
        })

        refresh_payload = {
            "sub": user.user_id,
            "type": "refresh",
            "exp": int(time.time()) + REFRESH_TOKEN_EXPIRY_SECONDS,
        }
        refresh_token = self._jwt.encode(refresh_payload)
        self._refresh_tokens[refresh_token] = {
            "user_id": user.user_id,
            "created_at": time.time(),
        }

        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": TOKEN_EXPIRY_SECONDS,
        }

    def verify_token(self, token: str) -> Dict:
        """Verify token and return claims."""
        if token in self._session_blacklist:
            raise InvalidTokenError("Token has been revoked", "TOKEN_REVOKED")
        return self._jwt.decode(token)

    def check_permission(self, token: str, required_permission: str) -> bool:
        """Check if token grants the required permission."""
        claims = self.verify_token(token)
        roles = [Role(r) for r in claims.get("roles", [])]
        granted = set()
        for role in roles:
            granted.update(ROLE_PERMISSIONS.get(role, []))
        return required_permission in granted

    def logout(self, token: str) -> Dict:
        self._session_blacklist.add(token)
        return {"success": True}
