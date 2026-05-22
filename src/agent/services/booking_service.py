from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from src.agent.services.booking_schemas import ErrorResponse, PaginatedResponse, SuccessResponse
from src.agent.services.http_service import JsonHttpServiceClient, ServiceApiError


DEFAULT_BOOKING_SERVICE_BASE_URL = "http://localhost:8081/api/v1"


class BookingServiceError(ServiceApiError):
    """Raised when the Booking Service returns an error or invalid response."""


class BookingServiceClient(JsonHttpServiceClient):
    error_cls = BookingServiceError

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(
            "Booking Service",
            base_url or os.getenv("BOOKING_SERVICE_BASE_URL") or DEFAULT_BOOKING_SERVICE_BASE_URL,
            http_client=http_client,
            timeout=timeout,
        )

    def hold_seats(self, *, showtime_id: str, seat_ids: list[str], user_id: str) -> dict[str, Any]:
        return self._validate_response(
            self.request_json(
                "POST",
                "/bookings/hold",
                json={
                    "seat_ids": seat_ids,
                    "showtime_id": showtime_id,
                    "user_id": user_id,
                },
            ),
            SuccessResponse,
        )

    def release_expired_holds(self) -> dict[str, Any]:
        return self._validate_response(
            self.request_json("POST", "/bookings/release-expired", json={}),
            SuccessResponse,
        )

    def get_bookings_by_user(self, *, user_id: str, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        clean_user_id = self._required_id(user_id, "user_id")
        return self._validate_response(
            self.get_json(
                f"/bookings/user/{quote(clean_user_id, safe='')}",
                params={"page": page, "page_size": page_size},
            ),
            PaginatedResponse,
        )

    def get_booking_by_id(self, booking_id: str) -> dict[str, Any]:
        clean_booking_id = self._required_id(booking_id, "booking_id")
        return self._validate_response(
            self.get_json(f"/bookings/{quote(clean_booking_id, safe='')}"),
            SuccessResponse,
        )

    def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        clean_booking_id = self._required_id(booking_id, "booking_id")
        return self._validate_response(
            self.request_json("POST", f"/bookings/{quote(clean_booking_id, safe='')}/cancel", json={}),
            SuccessResponse,
        )

    def confirm_booking(self, booking_id: str) -> dict[str, Any]:
        clean_booking_id = self._required_id(booking_id, "booking_id")
        return self._validate_response(
            self.request_json("POST", f"/bookings/{quote(clean_booking_id, safe='')}/confirm", json={}),
            SuccessResponse,
        )

    def get_showtime_seats(self, showtime_id: str) -> dict[str, Any]:
        clean_showtime_id = self._required_id(showtime_id, "showtime_id")
        return self._validate_response(
            self.get_json(f"/showtimes/{quote(clean_showtime_id, safe='')}/seats"),
            SuccessResponse,
        )

    def get_showtime_seats_ws_endpoint(self, showtime_id: str) -> dict[str, str]:
        clean_showtime_id = self._required_id(showtime_id, "showtime_id")
        http_url = self._url(f"/showtimes/{quote(clean_showtime_id, safe='')}/seats/ws")
        parsed = urlsplit(http_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return {
            "showtime_id": clean_showtime_id,
            "websocket_url": urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)),
            "message": "Use this WebSocket endpoint to stream realtime seat status updates.",
        }

    def _validate_response(
        self,
        payload: Any,
        schema: type[PaginatedResponse] | type[SuccessResponse],
    ) -> dict[str, Any]:
        try:
            return schema.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise BookingServiceError(f"Booking Service response does not match {schema.__name__}: {exc}") from exc

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = ErrorResponse.model_validate(payload)
        except ValueError:
            return f"Booking Service error {response.status_code}: {response.text}"
        except ValidationError:
            return f"Booking Service error {response.status_code}: {payload}"

        code = error.code if error.code is not None else response.status_code
        message = error.message or payload
        return f"Booking Service error {response.status_code}: code={code}, message={message}"

    @staticmethod
    def _required_id(value: str, field_name: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError(f"{field_name} is required.")
        return clean_value
