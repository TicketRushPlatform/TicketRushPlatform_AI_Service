from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)
_downstream_authorization: ContextVar[Optional[str]] = ContextVar(
    "downstream_authorization",
    default=None,
)


def set_downstream_authorization(authorization: Optional[str]) -> Token[Optional[str]]:
    return _downstream_authorization.set(authorization)


def reset_downstream_authorization(token: Token[Optional[str]]) -> None:
    _downstream_authorization.reset(token)


class ServiceApiError(RuntimeError):
    """Raised when a downstream service returns an error or invalid response."""


class JsonHttpServiceClient:
    error_cls = ServiceApiError

    def __init__(
        self,
        service_name: str,
        base_url: str,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 15.0,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_json(self, path: str, *, params: Optional[dict[str, Any]] = None) -> Any:
        return await self.request_json("GET", path, params=params)

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = self._url(path)
        logger.info(
            "downstream request service=%s method=%s url=%s params=%r json=%r",
            self.service_name,
            method,
            url,
            params,
            json,
        )
        try:
            response = await self._client.request(
                method,
                url,
                params=params,
                json=json,
                headers=self._headers(),
            )
            logger.info(
                "downstream response service=%s method=%s url=%s status_code=%s body=%r",
                self.service_name,
                method,
                url,
                response.status_code,
                response.text,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "downstream http status error service=%s method=%s url=%s status_code=%s",
                self.service_name,
                method,
                url,
                exc.response.status_code,
            )
            raise self.error_cls(self._error_message(exc.response)) from exc
        except httpx.RequestError as exc:
            logger.error(
                "downstream request error service=%s method=%s url=%s error=%s",
                self.service_name,
                method,
                url,
                exc,
            )
            raise self.error_cls(f"Cannot connect to {self.service_name}: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise self.error_cls(f"{self.service_name} returned non-JSON response.") from exc

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        authorization = _downstream_authorization.get()
        if authorization:
            headers["authorization"] = authorization
        return headers

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        return f"{self.service_name} error {response.status_code}: {payload}"
