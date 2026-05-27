import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agent.graph import (
    ainvoke_ticket_agent,
    build_model_node,
    build_ticket_agent,
    get_ticket_agent,
    invoke_ticket_agent,
    render_ticket_agent_prompt,
    route_after_model,
    sanitize_tool_message_history,
)
from src.agent.services.event_service import EventServiceClient
from src.agent.services.user_service import UserServiceClient
from src.agent.service_registry import (
    EVENT_SERVICE_PROVIDER,
    USER_SERVICE_PROVIDER,
    ServiceToolProvider,
    collect_service_tools,
)


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


class AgentServiceRegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_collect_service_tools_loads_event_tools_from_client_mapping(self):
        service_client = EventServiceClient(base_url="http://event-service.local/api/v1")

        tools = collect_service_tools(providers=[EVENT_SERVICE_PROVIDER], clients={"event": service_client})

        self.assertEqual(
            [tool.name for tool in tools],
            ["list_events", "get_event_by_id", "list_event_showtimes", "get_showtime_by_id"],
        )

    def test_collect_service_tools_loads_user_tools_from_client_mapping(self):
        service_client = UserServiceClient(base_url="http://user-service.local")

        tools = collect_service_tools(providers=[USER_SERVICE_PROVIDER], clients={"user": service_client})

        self.assertEqual(
            [tool.name for tool in tools],
            ["get_current_user", "list_users", "get_user_by_id", "get_user_stats"],
        )

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

    def test_get_ticket_agent_reuses_single_compiled_graph(self):
        class FakeCompiledGraph:
            pass

        fake_graph = FakeCompiledGraph()
        get_ticket_agent.cache_clear()
        try:
            with patch("src.agent.graph.build_ticket_agent", return_value=fake_graph) as build_agent:
                first = get_ticket_agent()
                second = get_ticket_agent()
        finally:
            get_ticket_agent.cache_clear()

        self.assertIs(first, fake_graph)
        self.assertIs(second, fake_graph)
        build_agent.assert_called_once_with()

    def test_route_after_model_goes_to_tools_only_when_model_requested_tool_calls(self):
        with_tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "get_booking_by_id", "args": {"booking_id": "b1"}, "id": "call_1"}],
        )
        without_tool_call = AIMessage(content="No tool needed.")

        self.assertEqual(route_after_model({"messages": [with_tool_call]}), "tools")
        self.assertEqual(route_after_model({"messages": [without_tool_call]}), END)

    async def test_model_node_logs_tool_calls_from_agent_response(self):
        class ToolCallingModel:
            def invoke(self, messages):
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "list_events", "args": {"page": 1, "page_size": 5}, "id": "call_1"}],
                )

        node = build_model_node(ToolCallingModel(), "TicketRush agent.")

        with self.assertLogs("src.agent.graph", level="INFO") as logs:
            await node({"messages": [{"role": "user", "content": "list events"}]})

        joined_logs = "\n".join(logs.output)
        self.assertIn("agent requested tool calls", joined_logs)
        self.assertIn("list_events", joined_logs)

    def test_sanitize_tool_message_history_removes_unanswered_tool_call_block(self):
        messages = [
            HumanMessage(content="list events"),
            AIMessage(
                content="",
                tool_calls=[{"name": "list_events", "args": {"page": 1}, "id": "call_1"}],
            ),
            HumanMessage(content="try again"),
        ]

        sanitized = sanitize_tool_message_history(messages)

        self.assertEqual([message.content for message in sanitized], ["list events", "try again"])

    def test_sanitize_tool_message_history_keeps_complete_tool_call_block(self):
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "list_events", "args": {"page": 1}, "id": "call_1"}],
        )
        tool_message = ToolMessage(content='{"data":[]}', tool_call_id="call_1")
        messages = [
            HumanMessage(content="list events"),
            ai_message,
            tool_message,
            AIMessage(content="No events found."),
        ]

        sanitized = sanitize_tool_message_history(messages)

        self.assertEqual(sanitized, messages)

    async def test_model_node_omits_unanswered_tool_call_history_before_model_invoke(self):
        class RecordingModel:
            def __init__(self):
                self.messages = None

            def invoke(self, messages):
                self.messages = messages
                return AIMessage(content="ok")

        model = RecordingModel()
        node = build_model_node(model, "TicketRush agent.")

        await node(
            {
                "messages": [
                    HumanMessage(content="list events"),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "list_events", "args": {"page": 1}, "id": "call_1"}],
                    ),
                    HumanMessage(content="try again"),
                ]
            }
        )

        self.assertEqual([message.content for message in model.messages], ["TicketRush agent.", "list events", "try again"])


class AsyncAgentInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ainvoke_ticket_agent_reuses_singleton_and_sets_thread_id(self):
        class FakeCompiledGraph:
            def __init__(self):
                self.calls = []

            async def ainvoke(self, state, *, config):
                self.calls.append((state, config))
                return {"messages": [AIMessage(content="ok")]}

        fake_graph = FakeCompiledGraph()
        get_ticket_agent.cache_clear()
        try:
            with patch("src.agent.graph.build_ticket_agent", return_value=fake_graph) as build_agent:
                result = await ainvoke_ticket_agent("hello", thread_id="user-1")
                await ainvoke_ticket_agent("again", thread_id="user-1")
        finally:
            get_ticket_agent.cache_clear()

        self.assertEqual(result["messages"][-1].content, "ok")
        self.assertEqual(len(fake_graph.calls), 2)
        self.assertEqual(fake_graph.calls[0][0], {"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(fake_graph.calls[0][1]["configurable"]["thread_id"], "user-1")
        build_agent.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
