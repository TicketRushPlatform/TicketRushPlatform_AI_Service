from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Optional


class InvalidAccessToken(ValueError):
    """Raised when a bearer access token cannot be trusted."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: str


_authenticated_user: ContextVar[Optional[AuthenticatedUser]] = ContextVar("authenticated_user", default=None)


def set_authenticated_user(user: Optional[AuthenticatedUser]) -> Token[Optional[AuthenticatedUser]]:
    return _authenticated_user.set(user)


def reset_authenticated_user(token: Token[Optional[AuthenticatedUser]]) -> None:
    _authenticated_user.reset(token)


def get_authenticated_user() -> Optional[AuthenticatedUser]:
    return _authenticated_user.get()


def require_authenticated_user() -> AuthenticatedUser:
    user = get_authenticated_user()
    if user is None:
        raise ValueError("An authenticated user is required for this booking operation.")
    return user


def scoped_thread_id(thread_id: str) -> str:
    clean_thread_id = thread_id.strip() or "default"
    user = get_authenticated_user()
    if user is None:
        return f"anon:{clean_thread_id}"
    return f"user:{user.user_id}:{clean_thread_id}"


def verify_bearer_access_token(authorization: str, *, secret: str, algorithm: str = "HS256") -> AuthenticatedUser:
    if not authorization.startswith("Bearer "):
        raise InvalidAccessToken("Bearer access token is required.")
    token = authorization.removeprefix("Bearer ").strip()
    payload = verify_access_token(token, secret=secret, algorithm=algorithm)
    return AuthenticatedUser(user_id=str(payload["sub"]), role=str(payload.get("role") or "").upper())


def verify_access_token(token: str, *, secret: str, algorithm: str = "HS256") -> dict[str, Any]:
    if algorithm != "HS256":
        raise InvalidAccessToken("Access token algorithm is unsupported.")

    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidAccessToken("Access token is invalid or expired.")

    encoded_header, encoded_payload, encoded_signature = parts
    try:
        header = _decode_json(encoded_header)
        payload = _decode_json(encoded_payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidAccessToken("Access token is invalid or expired.") from exc

    if header.get("alg") != algorithm:
        raise InvalidAccessToken("Access token algorithm is unsupported.")

    unsigned = f"{encoded_header}.{encoded_payload}"
    expected_signature = _base64url_encode(hmac.new(secret.encode(), unsigned.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected_signature, encoded_signature):
        raise InvalidAccessToken("Access token is invalid or expired.")

    if payload.get("type") != "access":
        raise InvalidAccessToken("Token type is invalid.")
    if not payload.get("sub"):
        raise InvalidAccessToken("Token subject is invalid.")

    exp = payload.get("exp")
    if not isinstance(exp, int | float) or exp <= time.time():
        raise InvalidAccessToken("Access token is invalid or expired.")

    return payload


def _decode_json(value: str) -> dict[str, Any]:
    decoded = base64.urlsafe_b64decode(_pad_base64(value))
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("JWT part must decode to an object.")
    return payload


def _pad_base64(value: str) -> str:
    return value + "=" * (-len(value) % 4)


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
