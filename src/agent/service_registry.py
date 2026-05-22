from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agent.services.booking_service import BookingServiceClient
from src.agent.services.event_service import EventServiceClient
from src.agent.tools.booking_tools import create_booking_service_tools
from src.agent.tools.event_tools import create_event_service_tools


ClientFactory = Callable[[], Any]
ToolFactory = Callable[[Any], Sequence[BaseTool]]


@dataclass(frozen=True)
class ServiceToolProvider:
    name: str
    description: str
    create_client: ClientFactory
    create_tools: ToolFactory


EVENT_SERVICE_PROVIDER = ServiceToolProvider(
    name="event",
    description="Event Service APIs for listing events/movies and reading event details.",
    create_client=EventServiceClient,
    create_tools=create_event_service_tools,
)

BOOKING_SERVICE_PROVIDER = ServiceToolProvider(
    name="booking",
    description="Booking Service APIs for seat holds, booking workflows, and showtime seat status.",
    create_client=BookingServiceClient,
    create_tools=create_booking_service_tools,
)

DEFAULT_SERVICE_PROVIDERS: tuple[ServiceToolProvider, ...] = (
    EVENT_SERVICE_PROVIDER,
    BOOKING_SERVICE_PROVIDER,
)


def collect_service_tools(
    *,
    providers: Sequence[ServiceToolProvider] = DEFAULT_SERVICE_PROVIDERS,
    clients: Optional[Mapping[str, Any]] = None,
) -> list[BaseTool]:
    tools: list[BaseTool] = []
    service_clients = clients or {}

    for provider in providers:
        client = service_clients.get(provider.name)
        if client is None:
            client = provider.create_client()
        tools.extend(provider.create_tools(client))

    return tools
