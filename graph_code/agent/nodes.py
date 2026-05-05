"""Node implementations for the LangGraph coding agent."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as lc_tool
from langgraph.types import interrupt

from ..config import Config, get_config
from ..llm.client import get_llm
from ..tools.interaction import get_interaction_store
from ..tools.permissions import (
    PermissionMode,
    build_permission_request,
    evaluate_permission,
)
from ..tools.runtime import ToolExecutionRuntime
from ..tools.schema import ToolResultEnvelope
from ..utils.debug import log_tool_execution
from .state import AgentState


SYSTEM_PROMPT = """You are Graph Code, a Claude Code-like coding agent built on LangGraph.
Work inside the configured project directory. Prefer reading code before editing it.
Use tools for filesystem, shell, tasks, memory, background work, schedules, teammates,
worktrees, skills, and MCP integrations. Keep final responses concise and factual."""


def _runtime(config: Config | None = None) -> ToolExecutionRuntime:
    cfg = config or get_config()
    return ToolExecutionRuntime(cfg.working_path, output_limit=cfg.output_limit)


def get_tools() -> list[StructuredTool]:
    """Return model-visible tools. Execution still goes through ToolExecutionRuntime."""

    @lc_tool
    def read_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
        """Read a file from the workspace."""
        return ""

    @lc_tool
    def write_file(file_path: str, content: str, append: bool = False) -> str:
        """Write or append a file in the workspace."""
        return ""

    @lc_tool
    def edit_file(file_path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace text in a file."""
        return ""

    @lc_tool
    def bash(command: str, timeout: int = 60) -> str:
        """Run a shell command in the workspace."""
        return ""

    @lc_tool
    def search_files(pattern: str, path: str = ".", glob: str | None = None) -> str:
        """Search files using a regex pattern."""
        return ""

    @lc_tool
    def todo(action: str = "list", items: list[dict[str, Any]] | None = None) -> str:
        """Read or set the todo list."""
        return ""

    @lc_tool
    def load_skill(name: str, path: str | None = None) -> str:
        """Load a skill body by name or path."""
        return ""

    @lc_tool
    def compact(mode: str = "manual", summary: str | None = None) -> str:
        """Request context compaction."""
        return ""

    @lc_tool
    def save_memory(namespace: str, key: str, value: str) -> str:
        """Save long-term memory under namespace/key."""
        return ""

    @lc_tool
    def task_create(subject: str, description: str = "", blocked_by: list[str] | None = None) -> str:
        """Create a persistent task."""
        return ""

    @lc_tool
    def task_update(task_id: str, status: str | None = None, owner: str | None = None) -> str:
        """Update a persistent task."""
        return ""

    @lc_tool
    def task_get(task_id: str) -> str:
        """Read a task."""
        return ""

    @lc_tool
    def task_list() -> str:
        """List tasks."""
        return ""

    @lc_tool
    def task_complete(task_id: str) -> str:
        """Complete a task and unblock dependents."""
        return ""

    @lc_tool
    def background_run(command: str, timeout: int = 3600) -> str:
        """Start a background command."""
        return ""

    @lc_tool
    def background_check(runtime_task_id: str) -> str:
        """Check a background command."""
        return ""

    @lc_tool
    def schedule_create(cron: str, prompt: str, recurring: bool = True, durable: bool = True) -> str:
        """Create a schedule that enqueues notifications when due."""
        return ""

    @lc_tool
    def schedule_list() -> str:
        """List due schedules as notifications."""
        return ""

    @lc_tool
    def schedule_delete(schedule_id: str) -> str:
        """Delete a schedule."""
        return ""

    @lc_tool
    def team_spawn(name: str, role: str = "worker", prompt: str = "") -> str:
        """Spawn a teammate/subagent record."""
        return ""

    @lc_tool
    def send_message(teammate_id: str, content: str, request_id: str | None = None) -> str:
        """Send protocol message to a teammate inbox."""
        return ""

    @lc_tool
    def request_shutdown(teammate_id: str, reason: str = "") -> str:
        """Request teammate shutdown."""
        return ""

    @lc_tool
    def submit_plan_approval(request_id: str, approved: bool, notes: str = "") -> str:
        """Submit a plan approval response."""
        return ""

    @lc_tool
    def claim_task(task_id: str, owner: str) -> str:
        """Claim a task with a lock."""
        return ""

    @lc_tool
    def worktree_create(task_id: str, base_path: str | None = None) -> str:
        """Create a worktree registry entry."""
        return ""

    @lc_tool
    def worktree_enter(worktree_id: str) -> str:
        """Enter a registered worktree."""
        return ""

    @lc_tool
    def worktree_run(worktree_id: str, command: str, timeout: int = 60) -> str:
        """Run a command in a worktree."""
        return ""

    @lc_tool
    def worktree_closeout(worktree_id: str, mode: str = "keep") -> str:
        """Close out a worktree, optionally removing it if clean."""
        return ""

    return [
        read_file,
        write_file,
        edit_file,
        bash,
        search_files,
        todo,
        load_skill,
        compact,
        save_memory,
        task_create,
        task_update,
        task_get,
        task_list,
        task_complete,
        background_run,
        background_check,
        schedule_create,
        schedule_list,
        schedule_delete,
        team_spawn,
        send_message,
        request_shutdown,
        submit_plan_approval,
        claim_task,
        worktree_create,
        worktree_enter,
        worktree_run,
        worktree_closeout,
    ]


def drain_notifications(state: AgentState) -> dict[str, Any]:
    if state.get("notifications"):
        return {"transition_reason": "notifications_drained"}
    return {"transition_reason": "no_notifications"}


def build_prompt(state: AgentState) -> dict[str, Any]:
    return {"transition_reason": "prompt_built"}


def call_model(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    """Call the configured model unless pending tool calls already exist."""
    pending = state.get("pending_tool_calls") or state.get("tool_calls")
    if pending:
        return {
            "pending_tool_calls": pending,
            "tool_calls": pending,
            "transition_reason": "pending_tools_reused",
        }

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state.get("messages", [])
    _sanitize_messages_for_utf8(messages)

    cfg = config or get_config()
    if cfg.llm_model == "mock":
        content = _mock_response_content(state)
        response = AIMessage(content=content)
    else:
        llm_with_tools = get_llm(config=cfg).bind_tools(get_tools())
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            return {
                "error": str(exc),
                "transition_reason": "model_error",
                "recovery_state": _consume_recovery_budget(state, "transient_retry_budget"),
            }

    _sanitize_message_for_utf8(response)
    if getattr(response, "tool_calls", None):
        return {
            "messages": [response],
            "pending_tool_calls": response.tool_calls,
            "tool_calls": response.tool_calls,
            "transition_reason": "model_requested_tools",
        }

    return {
        "messages": [response],
        "final_response": response.content,
        "transition_reason": "model_final_response",
    }


def route_model_response(state: AgentState) -> str:
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "tools"
    if state.get("error"):
        return "final"
    return "final" if state.get("final_response") else "final"


def permission_gate(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    mode = state.get("permission_mode") or (config or get_config()).permission_mode
    pending = list(state.get("pending_tool_calls") or state.get("tool_calls") or [])
    allowed: list[dict[str, Any]] = []
    denied: list[dict[str, Any]] = []

    for index, tool_call in enumerate(pending):
        decision = evaluate_permission(tool_call, mode)
        if decision.denied:
            denied.append(
                ToolResultEnvelope.error(
                    f"Tool blocked: {decision.reason}",
                    tool_call_id=(tool_call.get("id") or "unknown").strip() or "unknown",
                    metadata={"tool_name": tool_call.get("name", "unknown"), "permission": "denied"},
                ).model_dump()
            )
            continue
        if decision.ask:
            return {
                "pending_permission_request": build_permission_request(tool_call, decision),
                "pending_tool_calls": pending[index:],
                "tool_calls": pending[index:],
                "tool_results": denied,
                "transition_reason": "permission_interrupt_required",
            }
        allowed.append(tool_call)

    return {
        "pending_tool_calls": allowed,
        "tool_calls": allowed,
        "tool_results": denied,
        "pending_permission_request": None,
        "transition_reason": "permission_checked",
    }


def route_permission(state: AgentState) -> str:
    if state.get("pending_permission_request"):
        return "interrupt"
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "execute"
    if state.get("tool_results"):
        return "append"
    return "final"


def human_permission_interrupt(state: AgentState) -> dict[str, Any]:
    request = state.get("pending_permission_request")
    if not request:
        return {}
    resume = interrupt(request)
    approved = bool(resume.get("approved") if isinstance(resume, dict) else resume)
    reason = resume.get("reason", "") if isinstance(resume, dict) else ""
    call = request["tool_call"]
    remaining = list(state.get("pending_tool_calls") or [])
    if approved:
        return {
            "pending_permission_request": None,
            "pending_tool_calls": remaining,
            "tool_calls": remaining,
            "transition_reason": "permission_approved",
        }
    denied = ToolResultEnvelope.error(
        f"Permission denied: {reason or request.get('reason', '')}",
        tool_call_id=request.get("tool_call_id", call.get("id", "unknown")),
        metadata={"tool_name": call.get("name", "unknown"), "permission": "denied"},
    ).model_dump()
    return {
        "pending_permission_request": None,
        "pending_tool_calls": [],
        "tool_calls": [],
        "tool_results": list(state.get("tool_results") or []) + [denied],
        "transition_reason": "permission_denied",
    }


def route_after_human_permission(state: AgentState) -> str:
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "execute"
    return "append"


def run_pre_tool_hooks(state: AgentState) -> dict[str, Any]:
    return {"transition_reason": "pre_tool_hooks_complete"}


def execute_tools(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    calls = state.get("pending_tool_calls") or state.get("tool_calls") or []
    if not calls:
        return {}
    runtime = _runtime(config)
    results = runtime.execute(
        _sanitize_for_utf8(calls),
        permission_mode=state.get("permission_mode", PermissionMode.DEFAULT.value),
        skip_permissions=True,
    )
    for call, result in zip(calls, results):
        log_tool_execution(call.get("name", "unknown"), call.get("args", {}), result.content)
    merged = list(state.get("tool_results") or []) + [result.model_dump() for result in results]
    return {
        "pending_tool_calls": [],
        "tool_calls": [],
        "tool_results": merged,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "turn_count": state.get("turn_count", 0) + 1,
        "transition_reason": "tools_executed",
    }


def run_post_tool_hooks(state: AgentState) -> dict[str, Any]:
    return {"transition_reason": "post_tool_hooks_complete"}


def append_tool_results(state: AgentState) -> dict[str, Any]:
    messages: list[ToolMessage] = []
    for item in state.get("tool_results", []):
        envelope = item if isinstance(item, ToolResultEnvelope) else ToolResultEnvelope.model_validate(item)
        messages.append(
            ToolMessage(
                content=envelope.model_dump_json(),
                tool_call_id=envelope.tool_call_id or "unknown",
            )
        )
    return {
        "messages": messages,
        "tool_results": [],
        "transition_reason": "tool_results_appended",
    }


def compact_check(state: AgentState) -> dict[str, Any]:
    messages = state.get("messages", [])
    if len(messages) < 40:
        return {"transition_reason": "compact_not_needed"}
    summary = _summarize_messages(messages)
    compact_state = dict(state.get("compact_state") or {})
    compact_state.setdefault("summaries", []).append(summary)
    compact_state["mode"] = "summary"
    compact_state["last_compacted_turn"] = state.get("turn_count", 0)
    return {"compact_state": compact_state, "transition_reason": "summary_compact_complete"}


def recovery_handler(state: AgentState) -> dict[str, Any]:
    if state.get("error"):
        return {"transition_reason": "recovery_budget_recorded"}
    return {"transition_reason": state.get("transition_reason")}


def final_response(state: AgentState) -> dict[str, Any]:
    if state.get("final_response"):
        return {"final": True}
    if state.get("error"):
        return {"final_response": f"Error: {state['error']}", "final": True}
    last_ai = next((m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), None)
    return {"final_response": last_ai.content if last_ai else "", "final": True}


def agent_node(state: AgentState) -> dict[str, Any]:
    return call_model(state)


def tools_node(state: AgentState) -> dict[str, Any]:
    calls = state.get("pending_tool_calls") or state.get("tool_calls") or []
    if not calls:
        return {}
    runtime = ToolExecutionRuntime(Path.cwd())
    results = runtime.execute(_sanitize_for_utf8(calls), skip_permissions=True)
    messages = [
        ToolMessage(
            content=result.model_dump_json(),
            tool_call_id=result.tool_call_id or "unknown",
        )
        for result in results
    ]
    return {
        "tool_calls": [],
        "pending_tool_calls": [],
        "tool_results": messages,
        "messages": messages,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def check_interaction_node(state: AgentState) -> dict[str, Any]:
    store = get_interaction_store()
    if store.pending_question or store.pending_confirmation:
        return {
            "pending_question": bool(store.pending_question),
            "pending_confirmation": bool(store.pending_confirmation),
        }
    return {}


def handle_interaction_response(state: AgentState, user_input: str) -> dict[str, Any]:
    store = get_interaction_store()
    store.clear()
    return {
        "messages": [HumanMessage(content=user_input)],
        "pending_question": False,
        "pending_confirmation": False,
        "interaction_result": user_input,
    }


def should_continue(state: AgentState) -> str:
    if state.get("pending_question") or state.get("pending_confirmation"):
        return "pause"
    if state.get("final_response") or state.get("error"):
        return "end"
    if state.get("iteration_count", 0) >= get_config().max_tool_iterations:
        return "end"
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "execute_tools"
    return "end"


def _mock_response_content(state: AgentState) -> str:
    last_human = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return "Mock response."
    return f"Mock response: {last_human.content}"


def _consume_recovery_budget(state: AgentState, key: str) -> dict[str, Any]:
    recovery = dict(state.get("recovery_state") or {})
    recovery[key] = max(0, int(recovery.get(key, 0)) - 1)
    return recovery


def _summarize_messages(messages: list[Any]) -> dict[str, Any]:
    return {
        "current_goal": "Continue the user's coding task",
        "completed_actions": [getattr(m, "type", type(m).__name__) for m in messages[-8:]],
        "key_files": [],
        "key_decisions": [],
        "next_step": "Continue from the latest user request",
    }


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


def _sanitize_message_for_utf8(message: Any) -> Any:
    if hasattr(message, "content"):
        message.content = _sanitize_for_utf8(message.content)
    if hasattr(message, "additional_kwargs"):
        message.additional_kwargs = _sanitize_for_utf8(message.additional_kwargs)
    if hasattr(message, "response_metadata"):
        message.response_metadata = _sanitize_for_utf8(message.response_metadata)
    if hasattr(message, "tool_calls"):
        message.tool_calls = _sanitize_for_utf8(message.tool_calls)
    if hasattr(message, "invalid_tool_calls"):
        message.invalid_tool_calls = _sanitize_for_utf8(message.invalid_tool_calls)
    return message


def _sanitize_messages_for_utf8(messages: list[Any]) -> list[Any]:
    for message in messages:
        _sanitize_message_for_utf8(message)
    return messages
