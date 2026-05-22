from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.agent.services.booking_service import BookingServiceClient


class HoldBookingSeatsInput(BaseModel):
    showtime_id: str = Field(..., min_length=1, description="Showtime ID for POST /bookings/hold.")
    seat_ids: list[str] = Field(..., min_length=1, description="Seat IDs to hold. Must include at least one seat.")
    user_id: str = Field(..., min_length=1, description="User ID for the holding booking.")


class ReleaseExpiredBookingHoldsInput(BaseModel):
    pass


class GetBookingsByUserInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID path parameter for GET /bookings/user/{user_id}.")
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
            "Requires showtime_id, user_id, and one or more seat_ids."
        ),
    )
    def hold_booking_seats(showtime_id: str, seat_ids: list[str], user_id: str) -> Any:
        """Hold selected seats for a user and showtime."""
        return service_client.hold_seats(showtime_id=showtime_id, seat_ids=seat_ids, user_id=user_id)

    @tool(
        "release_expired_booking_holds",
        args_schema=ReleaseExpiredBookingHoldsInput,
        description="Call Booking Service POST /bookings/release-expired to release all expired holding bookings.",
    )
    def release_expired_booking_holds() -> Any:
        """Release expired holding bookings and seats."""
        return service_client.release_expired_holds()

    @tool(
        "get_bookings_by_user",
        args_schema=GetBookingsByUserInput,
        description=(
            "Call Booking Service GET /bookings/user/{user_id} to retrieve paginated bookings for a user. "
            "Use page and page_size when the user asks for a specific page."
        ),
    )
    def get_bookings_by_user(user_id: str, page: int = 1, page_size: int = 20) -> Any:
        """Get paginated bookings for a user."""
        return service_client.get_bookings_by_user(user_id=user_id, page=page, page_size=page_size)

    @tool(
        "get_booking_by_id",
        args_schema=BookingIdInput,
        description="Call Booking Service GET /bookings/{id} to retrieve booking details by booking ID.",
    )
    def get_booking_by_id(booking_id: str) -> Any:
        """Get booking details by ID."""
        return service_client.get_booking_by_id(booking_id)

    @tool(
        "cancel_booking",
        args_schema=BookingIdInput,
        description="Call Booking Service POST /bookings/{id}/cancel to cancel a booking and release held seats.",
    )
    def cancel_booking(booking_id: str) -> Any:
        """Cancel a booking."""
        return service_client.cancel_booking(booking_id)

    @tool(
        "confirm_booking",
        args_schema=BookingIdInput,
        description="Call Booking Service POST /bookings/{id}/confirm to confirm a held booking and mark seats sold.",
    )
    def confirm_booking(booking_id: str) -> Any:
        """Confirm a held booking."""
        return service_client.confirm_booking(booking_id)

    @tool(
        "get_showtime_seats",
        args_schema=ShowtimeIdInput,
        description="Call Booking Service GET /showtimes/{showtime_id}/seats to get seat status for a showtime.",
    )
    def get_showtime_seats(showtime_id: str) -> Any:
        """Get seat status for a showtime."""
        return service_client.get_showtime_seats(showtime_id)

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
