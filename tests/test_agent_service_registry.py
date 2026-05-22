import unittest

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agent.graph import build_ticket_agent, invoke_ticket_agent, render_ticket_agent_prompt, route_after_model
from src.agent.services.event_service import EventServiceClient
from src.agent.service_registry import EVENT_SERVICE_PROVIDER, ServiceToolProvider, collect_service_tools


class GetBookingByIdInput(BaseModel):
    booking_id: str = Field(..., min_length=1, description="Booking ID path parameter.")


class FakeChatModel:
    def __init__(self) -> None:
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="ok")


class AgentServiceRegistryTests(unittest.TestCase):
    def test_collect_service_tools_loads_event_tools_from_client_mapping(self):
        service_client = EventServiceClient(base_url="http://event-service.local/api/v1")

        tools = collect_service_tools(providers=[EVENT_SERVICE_PROVIDER], clients={"event": service_client})

        self.assertEqual([tool.name for tool in tools], ["list_events", "get_event_by_id"])

    def test_collect_service_tools_accepts_new_service_provider_without_graph_changes(self):
        @tool(
            "get_booking_by_id",
            args_schema=GetBookingByIdInput,
            description="Call Booking Service GET /bookings/{id}.",
        )
        def get_booking_by_id(booking_id: str) -> dict[str, str]:
            """Get booking details by ID."""
            return {"id": booking_id}

        provider = ServiceToolProvider(
            name="booking",
            description="Booking Service APIs.",
            create_client=lambda: object(),
            create_tools=lambda client: [get_booking_by_id],
        )

        tools = collect_service_tools(providers=[provider])

        self.assertEqual([tool.name for tool in tools], ["get_booking_by_id"])

    def test_render_ticket_agent_prompt_includes_dynamic_tool_catalog(self):
        @tool(
            "get_booking_by_id",
            args_schema=GetBookingByIdInput,
            description="Call Booking Service GET /bookings/{id}.",
        )
        def get_booking_by_id(booking_id: str) -> dict[str, str]:
            """Get booking details by ID."""
            return {"id": booking_id}

        prompt = render_ticket_agent_prompt(
            "TicketRush agent.\n\nAvailable service tools:\n{tool_catalog}",
            [get_booking_by_id],
        )

        self.assertIn("get_booking_by_id", prompt)
        self.assertIn("GET /bookings/{id}", prompt)

    def test_build_ticket_agent_creates_manual_graph_with_tool_node_edges(self):
        model = FakeChatModel()

        agent = build_ticket_agent(model=model, tools=[])
        graph = agent.get_graph()
        edges = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}

        self.assertIn("agent", graph.nodes)
        self.assertIn("tools", graph.nodes)
        self.assertIsInstance(graph.nodes["tools"].data, ToolNode)
        self.assertIn((START, "agent", False), edges)
        self.assertIn(("agent", "tools", True), edges)
        self.assertIn(("agent", END, True), edges)
        self.assertIn(("tools", "agent", False), edges)

    def test_build_ticket_agent_uses_in_memory_checkpointer_by_default(self):
        agent = build_ticket_agent(model=FakeChatModel(), tools=[])

        self.assertIsInstance(agent.checkpointer, InMemorySaver)

    def test_build_ticket_agent_accepts_explicit_checkpointer(self):
        checkpointer = InMemorySaver()

        agent = build_ticket_agent(model=FakeChatModel(), tools=[], checkpointer=checkpointer)

        self.assertIs(agent.checkpointer, checkpointer)

    def test_invoke_ticket_agent_sets_default_thread_id_for_checkpointing(self):
        result = invoke_ticket_agent("hello", model=FakeChatModel(), tools=[])

        self.assertEqual(result["messages"][-1].content, "ok")

    def test_route_after_model_goes_to_tools_only_when_model_requested_tool_calls(self):
        with_tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "get_booking_by_id", "args": {"booking_id": "b1"}, "id": "call_1"}],
        )
        without_tool_call = AIMessage(content="No tool needed.")

        self.assertEqual(route_after_model({"messages": [with_tool_call]}), "tools")
        self.assertEqual(route_after_model({"messages": [without_tool_call]}), END)


if __name__ == "__main__":
    unittest.main()
