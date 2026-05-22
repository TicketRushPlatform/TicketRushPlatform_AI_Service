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
    def list_events(
        page: int = 1,
        page_size: int = 20,
        event_type: Optional[Literal["EVENT", "MOVIE"]] = None,
        search: Optional[str] = None,
    ) -> Any:
        """List events from Event Service."""
        return service_client.list_events(
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
            "Use this only when the user provides a concrete event_id."
        ),
    )
    def get_event_by_id(event_id: str) -> Any:
        """Get one event detail from Event Service."""
        return service_client.get_event_by_id(event_id)

    return [list_events, get_event_by_id]
