from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.agent.services.event_service import EventServiceClient


class ListEventsInput(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number for Event Service pagination.")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of events to return per page.")
    event_type: Optional[Literal["EVENT", "MOVIE"]] = Field(
        default=None,
        description="Optional Event Service type filter. Use EVENT for normal events or MOVIE for movies.",
    )
    search: Optional[str] = Field(default=None, description="Optional event name search text.")


class GetEventByIdInput(BaseModel):
    event_id: str = Field(..., min_length=1, description="Event ID path parameter for GET /events/{id}.")


class ListEventShowtimesInput(BaseModel):
    event_id: str = Field(
        ...,
        min_length=1,
        description="Event ID path parameter for GET /events/{id}/showtimes. This is not a showtime ID.",
    )


class GetShowtimeByIdInput(BaseModel):
    showtime_id: str = Field(
        ...,
        min_length=1,
        description="Showtime ID path parameter for GET /showtimes/{id}. This is not an event ID.",
    )


def create_event_service_tools(client: Optional[EventServiceClient] = None) -> list[BaseTool]:
    service_client = client or EventServiceClient()

    @tool(
        "list_events",
        args_schema=ListEventsInput,
        description=(
            "Call Event Service GET /events to list events or movies. "
            "Use optional filters page, page_size, event_type, and search when the user asks to browse or search."
        ),
    )
    async def list_events(
        page: int = 1,
        page_size: int = 20,
        event_type: Optional[Literal["EVENT", "MOVIE"]] = None,
        search: Optional[str] = None,
    ) -> Any:
        """List events from Event Service."""
        return await service_client.list_events(
            page=page,
            page_size=page_size,
            event_type=event_type,
            search=search,
        )

    @tool(
        "get_event_by_id",
        args_schema=GetEventByIdInput,
        description=(
            "Call Event Service GET /events/{id} to retrieve one event or movie by ID. "
            "Use this only when the user provides a concrete event_id or after selecting one from list_events. "
            "This does not return every showtime for the event."
        ),
    )
    async def get_event_by_id(event_id: str) -> Any:
        """Get one event detail from Event Service."""
        return await service_client.get_event_by_id(event_id)

    @tool(
        "list_event_showtimes",
        args_schema=ListEventShowtimesInput,
        description=(
            "Call Event Service GET /events/{id}/showtimes to retrieve every showtime for one event. "
            "Use this when the user asks how many showtimes an event has, asks for dates/times, "
            "or wants to book seats for an event by name/date. Do not use GET /events/{id} for this."
        ),
    )
    async def list_event_showtimes(event_id: str) -> Any:
        """List all showtimes for one event."""
        return await service_client.list_showtimes_by_event(event_id)

    @tool(
        "get_showtime_by_id",
        args_schema=GetShowtimeByIdInput,
        description=(
            "Call Event Service GET /showtimes/{id} to retrieve one showtime by showtime_id. "
            "Use this only when the user provides a concrete showtime_id or after selecting one from list_event_showtimes."
        ),
    )
    async def get_showtime_by_id(showtime_id: str) -> Any:
        """Get one showtime detail from Event Service."""
        return await service_client.get_showtime_by_id(showtime_id)

    return [list_events, get_event_by_id, list_event_showtimes, get_showtime_by_id]
