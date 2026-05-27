from __future__ import annotations

import asyncio
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.services.event_service import EventServiceClient
from src.agent.service_registry import DEFAULT_SERVICE_PROVIDERS, ServiceToolProvider, collect_service_tools


PROMPT_PATH = Path(__file__).parent / "prompts" / "ticket_agent_system_prompt.md"
AGENT_NODE = "agent"
TOOLS_NODE = "tools"
logger = logging.getLogger(__name__)


def load_ticket_agent_prompt(prompt_path: Path = PROMPT_PATH) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def render_ticket_agent_prompt(prompt_template: str, tools: Sequence[BaseTool]) -> str:
    tool_catalog = "\n".join(f"- `{tool.name}`: {tool.description}" for tool in tools)
    if not tool_catalog:
        tool_catalog = "- No service tools are currently registered."

    return prompt_template.replace("{tool_catalog}", tool_catalog)


def route_after_model(state: MessagesState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return TOOLS_NODE

    if isinstance(last_message, dict) and last_message.get("tool_calls"):
        return TOOLS_NODE

    return END


def get_message_tool_call_ids(message: Any) -> list[str]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if not tool_calls:
        return []

    tool_call_ids = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("id")
        else:
            tool_call_id = getattr(tool_call, "id", None)
        if tool_call_id:
            tool_call_ids.append(str(tool_call_id))
    return tool_call_ids


def get_tool_response_call_id(message: Any) -> Optional[str]:
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id is None and isinstance(message, dict):
        tool_call_id = message.get("tool_call_id")
    return str(tool_call_id) if tool_call_id else None


def is_tool_response_message(message: Any) -> bool:
    if get_tool_response_call_id(message):
        return True
    if isinstance(message, dict):
        return message.get("role") == "tool"
    return getattr(message, "type", None) == "tool"


def sanitize_tool_message_history(messages: Sequence[Any]) -> list[Any]:
    sanitized: list[Any] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        expected_tool_call_ids = set(get_message_tool_call_ids(message))
        if not expected_tool_call_ids:
            if not is_tool_response_message(message):
                sanitized.append(message)
            index += 1
            continue

        block = [message]
        remaining_tool_call_ids = set(expected_tool_call_ids)
        cursor = index + 1
        while cursor < len(messages) and is_tool_response_message(messages[cursor]):
            tool_call_id = get_tool_response_call_id(messages[cursor])
            if tool_call_id in remaining_tool_call_ids:
                remaining_tool_call_ids.remove(tool_call_id)
            block.append(messages[cursor])
            cursor += 1

        if remaining_tool_call_ids:
            logger.warning(
                "dropping incomplete tool-call message block missing_tool_call_ids=%s",
                sorted(remaining_tool_call_ids),
            )
        else:
            sanitized.extend(block)
        index = cursor

    return sanitized


def build_model_node(model: Any, system_prompt: str) -> Callable[[MessagesState], Any]:
    async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=system_prompt), *sanitize_tool_message_history(state.get("messages", []))]
        logger.info("agent model invoked message_count=%d", len(messages))
        ainvoke = getattr(model, "ainvoke", None)
        if ainvoke is not None:
            response = await ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            logger.info("agent requested tool calls tool_calls=%r", tool_calls)
        else:
            logger.info("agent model response content=%r", getattr(response, "content", response))
        return {"messages": [response]}

    return call_model


def bind_tools_if_available(model: Any, tools: Sequence[BaseTool]) -> Any:
    if not tools:
        return model

    bind_tools = getattr(model, "bind_tools", None)
    if bind_tools is None:
        raise TypeError("Model must support bind_tools() when service tools are registered.")

    return bind_tools(tools)


def build_ticket_agent(
    *,
    event_service_client: Optional[EventServiceClient] = None,
    service_clients: Optional[Mapping[str, Any]] = None,
    service_providers: Sequence[ServiceToolProvider] = DEFAULT_SERVICE_PROVIDERS,
    tools: Optional[Sequence[BaseTool]] = None,
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    prompt_path: Path = PROMPT_PATH,
    checkpointer: Optional[Any] = None,
) -> Any:
    load_dotenv()
    resolved_clients = dict(service_clients or {})
    if event_service_client is not None:
        resolved_clients["event"] = event_service_client

    agent_tools = list(
        tools
        if tools is not None
        else collect_service_tools(
            providers=service_providers,
            clients=resolved_clients,
        )
    )
    chat_model = model or ChatOpenAI(
        model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    system_prompt = render_ticket_agent_prompt(load_ticket_agent_prompt(prompt_path), agent_tools)
    model_with_tools = bind_tools_if_available(chat_model, agent_tools)

    graph = StateGraph(MessagesState)
    graph.add_node(AGENT_NODE, build_model_node(model_with_tools, system_prompt))
    graph.add_node(TOOLS_NODE, ToolNode(agent_tools))
    graph.add_edge(START, AGENT_NODE)
    graph.add_conditional_edges(
        AGENT_NODE,
        route_after_model,
        {
            TOOLS_NODE: TOOLS_NODE,
            END: END,
        },
    )
    graph.add_edge(TOOLS_NODE, AGENT_NODE)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache(maxsize=1)
def get_ticket_agent() -> Any:
    return build_ticket_agent()


async def ainvoke_ticket_agent(
    message: str,
    *,
    thread_id: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    agent = build_ticket_agent(**kwargs) if kwargs else get_ticket_agent()
    invoke_config = dict(config or {})
    configurable = dict(invoke_config.get("configurable") or {})
    configurable.setdefault("thread_id", thread_id)
    invoke_config["configurable"] = configurable

    ainvoke = getattr(agent, "ainvoke", None)
    if ainvoke is not None:
        return await ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=invoke_config,
        )

    return await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": message}]},
        config=invoke_config,
    )


def invoke_ticket_agent(
    message: str,
    *,
    thread_id: str = "default",
    config: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    return asyncio.run(ainvoke_ticket_agent(message, thread_id=thread_id, config=config, **kwargs))


load_event_agent_prompt = load_ticket_agent_prompt
build_event_agent = build_ticket_agent
invoke_event_agent = invoke_ticket_agent
ainvoke_event_agent = ainvoke_ticket_agent
