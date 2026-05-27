from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.agent.auth_context import require_authenticated_user
from src.agent.services.booking_service import BookingServiceClient


class HoldBookingSeatsInput(BaseModel):
    showtime_id: str = Field(
        ...,
        min_length=1,
        description="Showtime ID for POST /bookings/hold. Never pass an event_id here.",
    )
    seat_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Seat UUIDs from GET /showtimes/{showtime_id}/seats. Do not pass display labels such as C1.",
    )


class ReleaseExpiredBookingHoldsInput(BaseModel):
    pass


class GetBookingsByUserInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="User ID path parameter for GET /bookings/user/{user_id}. Omit for the authenticated user.",
    )
    page: int = Field(default=1, ge=1, description="Page number for bookings pagination.")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of bookings to return per page.")


class BookingIdInput(BaseModel):
    booking_id: str = Field(..., min_length=1, description="Booking ID path parameter.")


class ShowtimeIdInput(BaseModel):
    showtime_id: str = Field(..., min_length=1, description="Showtime ID path parameter.")


def create_booking_service_tools(client: Optional[BookingServiceClient] = None) -> list[BaseTool]:
    service_client = client or BookingServiceClient()

    @tool(
        "hold_booking_seats",
        args_schema=HoldBookingSeatsInput,
        description=(
            "Call Booking Service POST /bookings/hold to create a holding booking for selected seats. "
            "Requires a showtime_id, not an event_id, and one or more seat UUIDs from get_showtime_seats. "
            "The user is always the authenticated user."
        ),
    )
    async def hold_booking_seats(showtime_id: str, seat_ids: list[str]) -> Any:
        """Hold selected seats for a user and showtime."""
        user_id = require_authenticated_user().user_id
        return await service_client.hold_seats(showtime_id=showtime_id, seat_ids=seat_ids, user_id=user_id)

    @tool(
        "release_expired_booking_holds",
        args_schema=ReleaseExpiredBookingHoldsInput,
        description="Call Booking Service POST /bookings/release-expired to release all expired holding bookings.",
    )
    async def release_expired_booking_holds() -> Any:
        """Release expired holding bookings and seats."""
        return await service_client.release_expired_holds()

    @tool(
        "get_bookings_by_user",
        args_schema=GetBookingsByUserInput,
        description=(
            "Call Booking Service GET /bookings/user/{user_id} to retrieve paginated bookings. "
            "For normal users, always use the authenticated user's ID. Admins may request another user_id."
        ),
    )
    async def get_bookings_by_user(user_id: Optional[str] = None, page: int = 1, page_size: int = 20) -> Any:
        """Get paginated bookings for a user."""
        user = require_authenticated_user()
        target_user_id = user_id if user.role == "ADMIN" and user_id else user.user_id
        return await service_client.get_bookings_by_user(user_id=target_user_id, page=page, page_size=page_size)

    @tool(
        "get_booking_by_id",
        args_schema=BookingIdInput,
        description="Call Booking Service GET /bookings/{id} to retrieve booking details by booking ID.",
    )
    async def get_booking_by_id(booking_id: str) -> Any:
        """Get booking details by ID."""
        return await service_client.get_booking_by_id(booking_id)

    @tool(
        "cancel_booking",
        args_schema=BookingIdInput,
        description="Call Booking Service POST /bookings/{id}/cancel to cancel a booking and release held seats.",
    )
    async def cancel_booking(booking_id: str) -> Any:
        """Cancel a booking."""
        return await service_client.cancel_booking(booking_id)

    @tool(
        "confirm_booking",
        args_schema=BookingIdInput,
        description="Call Booking Service POST /bookings/{id}/confirm to confirm a held booking and mark seats sold.",
    )
    async def confirm_booking(booking_id: str) -> Any:
        """Confirm a held booking."""
        return await service_client.confirm_booking(booking_id)

    @tool(
        "get_showtime_seats",
        args_schema=ShowtimeIdInput,
        description="Call Booking Service GET /showtimes/{showtime_id}/seats to get seat status for a showtime.",
    )
    async def get_showtime_seats(showtime_id: str) -> Any:
        """Get seat status for a showtime."""
        return await service_client.get_showtime_seats(showtime_id)

    @tool(
        "get_showtime_seats_ws_endpoint",
        args_schema=ShowtimeIdInput,
        description=(
            "Return the Booking Service WebSocket endpoint /showtimes/{showtime_id}/seats/ws for realtime seat updates. "
            "This endpoint is a persistent WebSocket stream, not a JSON HTTP response."
        ),
    )
    def get_showtime_seats_ws_endpoint(showtime_id: str) -> Any:
        """Get the WebSocket endpoint for realtime showtime seat status."""
        return service_client.get_showtime_seats_ws_endpoint(showtime_id)

    return [
        hold_booking_seats,
        release_expired_booking_holds,
        get_bookings_by_user,
        get_booking_by_id,
        cancel_booking,
        confirm_booking,
        get_showtime_seats,
        get_showtime_seats_ws_endpoint,
    ]
