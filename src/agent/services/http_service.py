from __future__ import annotations

from typing import Any, Optional

import httpx


class ServiceApiError(RuntimeError):
    """Raised when a downstream service returns an error or invalid response."""


class JsonHttpServiceClient:
    error_cls = ServiceApiError

    def __init__(
        self,
        service_name: str,
        base_url: str,
        *,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get_json(self, path: str, *, params: Optional[dict[str, Any]] = None) -> Any:
        return self.request_json("GET", path, params=params)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                self._url(path),
                params=params,
                json=json,
                headers={"accept": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self.error_cls(self._error_message(exc.response)) from exc
        except httpx.RequestError as exc:
            raise self.error_cls(f"Cannot connect to {self.service_name}: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise self.error_cls(f"{self.service_name} returned non-JSON response.") from exc

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        return f"{self.service_name} error {response.status_code}: {payload}"
