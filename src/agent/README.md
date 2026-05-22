# TicketRush Agent Architecture

This package uses one LangChain/LangGraph-backed agent. Do not create a new agent per service.

## Layout

```text
src/agent/
  graph.py                  # Builds the single TicketRush agent with StateGraph nodes/edges
  service_registry.py       # Registers service tool providers
  prompts/                  # Agent-level prompts
  services/                 # HTTP clients and response schemas for downstream services
  tools/                    # @tool functions and schemas for service APIs
```

## Add a New Service

1. Create a service client, for example `services/booking_service.py`.
2. Create one `@tool(args_schema=...)` per API, for example in `tools/booking_tools.py`.
3. Add response schemas for service responses when the API has a documented shape.
4. Register one `ServiceToolProvider` in `service_registry.py`.
5. Keep endpoint-specific behavior inside the tool descriptions and schemas.
6. Keep shared orchestration rules in `prompts/ticket_agent_system_prompt.md`.

The graph is built manually with `StateGraph`, an `agent` model node, a `tools` `ToolNode`, and conditional edges based on model tool calls. It loads all registered providers through `collect_service_tools()`, so `graph.py` should rarely change when adding services.

The compiled graph uses LangGraph's in-memory checkpointer by default. Pass a stable `thread_id` when invoking to keep conversation state separated per user/session.

## Service URLs

- Event Service: `EVENT_SERVICE_BASE_URL`, default `http://localhost:8080/api/v1`
- Booking Service: `BOOKING_SERVICE_BASE_URL`, default `http://localhost:8081/api/v1`
