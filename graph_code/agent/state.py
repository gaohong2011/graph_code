"""State definitions for the LangGraph coding agent."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Custom graph state.

    Messages use LangGraph's message reducer; the rest of the fields are
    explicit control-plane state for tools, approvals, tasks, memory, and
    recovery.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    transition_reason: str | None
    pending_tool_calls: list[dict[str, Any]]
    pending_permission_request: dict[str, Any] | None
    tool_results: list[dict[str, Any]]
    planning_state: dict[str, Any]
    compact_state: dict[str, Any]
    recovery_state: dict[str, Any]
    loaded_skills: dict[str, Any]
    notifications: list[dict[str, Any]]
    runtime_tasks: dict[str, Any]
    current_task_id: str | None
    teammate_identity: dict[str, Any] | None
    worktree_context: dict[str, Any]
    mcp_connection_state: dict[str, Any]

    # Compatibility fields retained for the original public API/tests.
    current_task: str | None
    tool_calls: list[dict[str, Any]]
    iteration_count: int
    pending_confirmation: bool
    pending_question: bool
    interaction_result: str | None
    final_response: str | None
    error: str | None

    # Runtime knobs.
    permission_mode: str
    final: NotRequired[bool]


GraphCodeState = AgentState


def create_initial_state(permission_mode: str = "default") -> AgentState:
    """Create a fresh state with all required custom fields initialized."""
    return {
        "messages": [],
        "turn_count": 0,
        "transition_reason": None,
        "pending_tool_calls": [],
        "pending_permission_request": None,
        "tool_results": [],
        "planning_state": {"status": "none", "approved": False},
        "compact_state": {"mode": "none", "summaries": [], "last_compacted_turn": 0},
        "recovery_state": {
            "max_tokens_budget": 2,
            "context_retry_budget": 1,
            "transient_retry_budget": 2,
        },
        "loaded_skills": {},
        "notifications": [],
        "runtime_tasks": {},
        "current_task_id": None,
        "teammate_identity": None,
        "worktree_context": {"current": None, "registry": {}},
        "mcp_connection_state": {},
        "current_task": None,
        "tool_calls": [],
        "iteration_count": 0,
        "pending_confirmation": False,
        "pending_question": False,
        "interaction_result": None,
        "final_response": None,
        "error": None,
        "permission_mode": permission_mode,
    }
