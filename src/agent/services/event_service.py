from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from src.agent.services.event_schemas import ErrorResponse, PaginatedResponse, SuccessResponse
from src.agent.services.http_service import JsonHttpServiceClient, ServiceApiError


DEFAULT_EVENT_SERVICE_BASE_URL = "http://localhost:8080/api/v1"


class EventServiceError(ServiceApiError):
    """Raised when the Event Service returns an error or invalid response."""


class EventServiceClient(JsonHttpServiceClient):
    error_cls = EventServiceError

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(
            "Event Service",
            base_url or os.getenv("EVENT_SERVICE_BASE_URL") or DEFAULT_EVENT_SERVICE_BASE_URL,
            http_client=http_client,
            timeout=timeout,
        )

    def list_events(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        event_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Any:
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if event_type:
            params["type"] = event_type
        if search:
            params["search"] = search

        return self._validate_response(
            self.get_json("/events", params=params),
            PaginatedResponse,
        )

    def get_event_by_id(self, event_id: str) -> Any:
        clean_event_id = event_id.strip()
        if not clean_event_id:
            raise ValueError("event_id is required.")

        return self._validate_response(
            self.get_json(f"/events/{quote(clean_event_id, safe='')}"),
            SuccessResponse,
        )

    def _validate_response(self, payload: Any, schema: type[PaginatedResponse] | type[SuccessResponse]) -> dict[str, Any]:
        try:
            return schema.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise EventServiceError(f"Event Service response does not match {schema.__name__}: {exc}") from exc

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = ErrorResponse.model_validate(payload)
        except ValueError:
            return f"Event Service error {response.status_code}: {response.text}"
        except ValidationError:
            return f"Event Service error {response.status_code}: {payload}"

        code = error.code if error.code is not None else response.status_code
        message = error.message or payload
        return f"Event Service error {response.status_code}: code={code}, message={message}"
