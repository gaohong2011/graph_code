"""LangGraph builder and runners for Graph Code."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command

from ..config import Config, get_config
from .nodes import (
    append_tool_results,
    build_prompt,
    call_model,
    compact_check,
    drain_notifications,
    execute_tools,
    final_response,
    human_permission_interrupt,
    permission_gate,
    recovery_handler,
    route_after_human_permission,
    route_model_response,
    route_permission,
    run_post_tool_hooks,
    run_pre_tool_hooks,
)
from .persistence import create_checkpointer, create_store
from .session_memory.updater import maybe_update_session_memory
from .state import AgentState, create_initial_state


_CHECKPOINTER = None
_STORE = None


def build_agent(
    config: Config | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """Build and compile the coding-agent StateGraph."""
    cfg = config or get_config()
    workflow = StateGraph(AgentState)

    def call_model_node(state: AgentState) -> dict[str, Any]:
        return call_model(state, config=cfg)

    def build_prompt_node(state: AgentState) -> dict[str, Any]:
        return build_prompt(state, config=cfg)

    def recovery_handler_node(state: AgentState) -> dict[str, Any]:
        return recovery_handler(state, config=cfg)

    def permission_gate_node(state: AgentState) -> dict[str, Any]:
        return permission_gate(state, config=cfg)

    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        return execute_tools(state, config=cfg)

    def compact_check_node(state: AgentState) -> dict[str, Any]:
        return compact_check(state, config=cfg)

    def final_response_node(state: AgentState) -> dict[str, Any]:
        update = final_response(state, config=cfg)
        merged = dict(state)
        merged.update(update)
        session_update = maybe_update_session_memory(merged, cfg)
        update.update(session_update)
        return update

    workflow.add_node("drain_notifications", drain_notifications)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("recovery_handler", recovery_handler_node)
    workflow.add_node("route_model_response", lambda state: {})
    workflow.add_node("permission_gate", permission_gate_node)
    workflow.add_node("human_permission_interrupt", human_permission_interrupt)
    workflow.add_node("run_pre_tool_hooks", run_pre_tool_hooks)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("run_post_tool_hooks", run_post_tool_hooks)
    workflow.add_node("append_tool_results", append_tool_results)
    workflow.add_node("compact_check", compact_check_node)
    workflow.add_node("recovery_handler_after_tools", recovery_handler_node)
    workflow.add_node("final_response", final_response_node)

    workflow.add_edge(START, "drain_notifications")
    workflow.add_edge("drain_notifications", "build_prompt")
    workflow.add_edge("build_prompt", "call_model")
    workflow.add_edge("call_model", "recovery_handler")
    workflow.add_edge("recovery_handler", "route_model_response")
    workflow.add_conditional_edges(
        "route_model_response",
        route_model_response,
        {
            "retry": "call_model",
            "tools": "permission_gate",
            "final": "final_response",
        },
    )
    workflow.add_conditional_edges(
        "permission_gate",
        route_permission,
        {
            "interrupt": "human_permission_interrupt",
            "execute": "run_pre_tool_hooks",
            "append": "append_tool_results",
            "final": "final_response",
        },
    )
    workflow.add_conditional_edges(
        "human_permission_interrupt",
        route_after_human_permission,
        {
            "permission": "permission_gate",
            "execute": "run_pre_tool_hooks",
            "append": "append_tool_results",
        },
    )
    workflow.add_edge("run_pre_tool_hooks", "execute_tools")
    workflow.add_edge("execute_tools", "run_post_tool_hooks")
    workflow.add_edge("run_post_tool_hooks", "append_tool_results")
    workflow.add_edge("append_tool_results", "compact_check")
    workflow.add_edge("compact_check", "recovery_handler_after_tools")
    workflow.add_edge("recovery_handler_after_tools", "call_model")
    workflow.add_edge("final_response", END)

    return workflow.compile(
        checkpointer=checkpointer or _default_checkpointer(cfg),
        store=store or _default_store(cfg),
    )


def _default_checkpointer(config: Config):
    global _CHECKPOINTER
    if config.checkpoint_backend != "memory":
        return create_checkpointer(config)
    if _CHECKPOINTER is None:
        _CHECKPOINTER = create_checkpointer(config)
    return _CHECKPOINTER


def _default_store(config: Config):
    global _STORE
    if _STORE is None:
        _STORE = create_store(config)
    if config.store_backend != "memory":
        return create_store(config)
    return _STORE


def run_agent(
    user_input: str,
    state: AgentState | None = None,
    thread_id: str | None = None,
    config: Config | None = None,
    stream_mode: str | list[str] = "updates",
) -> Iterator[Any]:
    """Stream one user turn through the graph."""
    cfg = config or get_config()
    if state is None:
        state = create_initial_state(permission_mode=cfg.permission_mode)

    _prepare_turn_state(state, user_input)

    graph = build_agent(config=cfg)
    graph_config = {"configurable": {"thread_id": thread_id or "default"}}
    for event in graph.stream(state, graph_config, stream_mode=stream_mode):
        if stream_mode == "updates" and isinstance(event, dict):
            if "__interrupt__" in event:
                yield event
                continue
            for _node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    _sync_state_update(state, node_output)
                    yield node_output
        else:
            yield event


async def run_agent_async(
    user_input: str,
    state: AgentState | None = None,
    thread_id: str | None = None,
    config: Config | None = None,
    stream_mode: str | list[str] = "updates",
) -> AsyncIterator[Any]:
    cfg = config or get_config()
    if state is None:
        state = create_initial_state(permission_mode=cfg.permission_mode)
    _prepare_turn_state(state, user_input)
    graph = build_agent(config=cfg)
    graph_config = {"configurable": {"thread_id": thread_id or "default"}}
    async for event in graph.astream(state, graph_config, stream_mode=stream_mode):
        if stream_mode == "updates" and isinstance(event, dict):
            for node_output in event.values():
                if isinstance(node_output, dict):
                    _sync_state_update(state, node_output)
        yield event


def resume_with_interaction(
    state: AgentState,
    user_response: str,
    thread_id: str | None = None,
    config: Config | None = None,
) -> Iterator[Any]:
    """Compatibility helper for the old interaction-store flow."""
    from .nodes import handle_interaction_response

    update = handle_interaction_response(state, user_response)
    state["messages"].extend(update.get("messages", []))
    for key, value in update.items():
        if key != "messages":
            state[key] = value
    graph = build_agent(config=config)
    graph_config = {"configurable": {"thread_id": thread_id or "default"}}
    for event in graph.stream(state, graph_config, stream_mode="updates"):
        if isinstance(event, dict):
            for _node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    _sync_state_update(state, node_output)
                    yield node_output


def resume_graph(
    resume: dict[str, Any] | str | bool,
    thread_id: str,
    config: Config | None = None,
    state: AgentState | None = None,
) -> Iterator[Any]:
    """Resume a LangGraph interrupt with Command(resume=...)."""
    graph = build_agent(config=config)
    graph_config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream(Command(resume=resume), graph_config, stream_mode="updates"):
        if state is not None and isinstance(event, dict):
            for node_output in event.values():
                if isinstance(node_output, dict):
                    _sync_state_update(state, node_output)
        yield event


def _sync_state_update(state: AgentState, update: dict[str, Any]) -> None:
    """Keep caller-held state aligned with streamed LangGraph updates."""
    for key, value in update.items():
        if key not in state or key == "final_response":
            continue
        if key == "messages":
            state["messages"] = add_messages(state["messages"], value)
        else:
            state[key] = value


def _prepare_turn_state(state: AgentState, user_input: str) -> None:
    """Reset per-turn control state and append the new user message."""
    state["final_response"] = None
    state["error"] = None
    state["pending_question"] = False
    state["pending_confirmation"] = False
    state["pending_tool_calls"] = []
    state["approved_tool_calls"] = []
    state["tool_calls"] = []
    state["tool_results"] = []
    message = HumanMessage(content=user_input)
    state["messages"].append(message)
    if state.get("context_messages"):
        state["context_messages"].append(message)
