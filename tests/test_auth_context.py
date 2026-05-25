import base64
import hashlib
import hmac
import json
import time
import unittest

from src.agent.auth_context import (
    AuthenticatedUser,
    InvalidAccessToken,
    reset_authenticated_user,
    scoped_thread_id,
    set_authenticated_user,
    verify_bearer_access_token,
)


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def make_token(*, secret: str = "test-secret", sub: str = "user-1", role: str = "USER", token_type: str = "access") -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": sub,
        "role": role,
        "type": token_type,
        "exp": int(time.time()) + 3600,
    }
    unsigned = f"{_base64url(json.dumps(header, separators=(',', ':')).encode())}.{_base64url(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(secret.encode(), unsigned.encode(), hashlib.sha256).digest()
    return f"{unsigned}.{_base64url(signature)}"


class AuthContextTests(unittest.TestCase):
    def test_verify_bearer_access_token_returns_user_claims(self):
        token = make_token(sub="77777777-7777-7777-7777-777777777777", role="customer")

        user = verify_bearer_access_token(f"Bearer {token}", secret="test-secret", algorithm="HS256")

        self.assertEqual(user.user_id, "77777777-7777-7777-7777-777777777777")
        self.assertEqual(user.role, "CUSTOMER")

    def test_verify_bearer_access_token_rejects_wrong_signature(self):
        token = make_token(secret="wrong-secret")

        with self.assertRaises(InvalidAccessToken):
            verify_bearer_access_token(f"Bearer {token}", secret="test-secret", algorithm="HS256")

    def test_scoped_thread_id_uses_current_user_when_available(self):
        token = set_authenticated_user(AuthenticatedUser(user_id="user-1", role="USER"))
        try:
            self.assertEqual(scoped_thread_id("default"), "user:user-1:default")
        finally:
            reset_authenticated_user(token)

    def test_scoped_thread_id_separates_anonymous_sessions(self):
        self.assertEqual(scoped_thread_id("default"), "anon:default")


if __name__ == "__main__":
    unittest.main()
