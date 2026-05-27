from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from src.agent.services.http_service import JsonHttpServiceClient, ServiceApiError
from src.agent.services.user_schemas import ErrorResponse, UserResponse, UserStatsResponse


DEFAULT_USER_SERVICE_BASE_URL = "http://localhost:8082"


class UserServiceError(ServiceApiError):
    """Raised when the User Service returns an error or invalid response."""


class UserServiceClient(JsonHttpServiceClient):
    error_cls = UserServiceError

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(
            "User Service",
            base_url or os.getenv("USER_SERVICE_BASE_URL") or DEFAULT_USER_SERVICE_BASE_URL,
            http_client=http_client,
            timeout=timeout,
        )

    async def get_current_user(self) -> dict[str, Any]:
        payload = await self.get_json("/users/me")
        return self._validate_user(payload)

    async def list_users(self) -> list[dict[str, Any]]:
        payload = await self.get_json("/users")
        if not isinstance(payload, list):
            raise UserServiceError("User Service response does not match list[UserResponse].")
        return [self._validate_user(item) for item in payload]

    async def get_user_by_id(self, user_id: str) -> dict[str, Any]:
        clean_user_id = user_id.strip()
        if not clean_user_id:
            raise ValueError("user_id is required.")

        payload = await self.get_json(f"/users/{quote(clean_user_id, safe='')}")
        return self._validate_user(payload)

    async def get_user_stats(self) -> dict[str, Any]:
        payload = await self.get_json("/users/stats")
        try:
            return UserStatsResponse.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise UserServiceError(f"User Service response does not match UserStatsResponse: {exc}") from exc

    def _validate_user(self, payload: Any) -> dict[str, Any]:
        try:
            return UserResponse.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise UserServiceError(f"User Service response does not match UserResponse: {exc}") from exc

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = ErrorResponse.model_validate(payload)
        except ValueError:
            return f"User Service error {response.status_code}: {response.text}"
        except ValidationError:
            return f"User Service error {response.status_code}: {payload}"

        code = error.code if error.code is not None else response.status_code
        message = error.message or payload
        return f"User Service error {response.status_code}: code={code}, message={message}"
