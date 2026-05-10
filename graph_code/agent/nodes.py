"""Node implementations for the LangGraph coding agent."""

from __future__ import annotations

import time
import hashlib
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as lc_tool
from langgraph.types import interrupt

from ..config import Config, get_config
from ..llm.client import get_llm
from ..llm.protocol import validate_tool_message_protocol
from ..tools.interaction import get_interaction_store
from ..tools.permissions import (
    PermissionMode,
    build_permission_request,
    evaluate_permission,
)
from ..tools.runtime import ToolExecutionRuntime
from ..tools.schema import ToolResultEnvelope
from ..utils.debug import log_tool_execution
from .compaction import (
    CompactionOutput,
    compact_messages,
    format_summary,
    get_compaction_policy,
)
from .compaction.prompt import build_model_compact_prompt, format_model_compact_summary
from .compaction.runtime_context import (
    build_rehydration_text,
    run_compact_hook,
    write_transcript,
)
from .prompt.builder import build_system_prompt
from .prompt.cache import invalidate_prompt_cache
from .state import AgentState


SYSTEM_PROMPT = """You are Graph Code, a Claude Code-like coding agent built on LangGraph.
Work inside the configured project directory. Prefer reading code before editing it.
Use tools for filesystem, shell, tasks, memory, background work, schedules, teammates,
worktrees, skills, and MCP integrations. Keep final responses concise and factual."""


def _runtime(config: Config | None = None) -> ToolExecutionRuntime:
    cfg = config or get_config()
    return ToolExecutionRuntime(cfg.working_path, output_limit=cfg.output_limit, config=cfg)


def get_tools() -> list[StructuredTool]:
    """Return model-visible tools backed by the same runtime as graph execution."""

    @lc_tool
    def read_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
        """Read a file from the workspace."""
        return _invoke_schema_tool("read_file", file_path=file_path, offset=offset, limit=limit)

    @lc_tool
    def write_file(file_path: str, content: str, append: bool = False) -> str:
        """Write or append a file in the workspace."""
        return _invoke_schema_tool("write_file", file_path=file_path, content=content, append=append)

    @lc_tool
    def edit_file(file_path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace text in a file."""
        return _invoke_schema_tool(
            "edit_file",
            file_path=file_path,
            old=old,
            new=new,
            replace_all=replace_all,
        )

    @lc_tool
    def bash(command: str, timeout: int = 60) -> str:
        """Run a shell command in the workspace."""
        return _invoke_schema_tool("bash", command=command, timeout=timeout)

    @lc_tool
    def search_files(pattern: str, path: str = ".", glob: str | None = None) -> str:
        """Search files using a regex pattern."""
        return _invoke_schema_tool("search_files", pattern=pattern, path=path, glob=glob)

    @lc_tool
    def todo(action: str = "list", items: list[dict[str, Any]] | None = None) -> str:
        """Read or set the todo list."""
        return _invoke_schema_tool("todo", action=action, items=items)

    @lc_tool
    def load_skill(name: str, path: str | None = None) -> str:
        """Load a skill body by name or path."""
        return _invoke_schema_tool("load_skill", name=name, path=path)

    @lc_tool
    def compact(mode: str = "manual", summary: str | None = None) -> str:
        """Request context compaction."""
        return _invoke_schema_tool("compact", mode=mode, summary=summary)

    @lc_tool
    def save_memory(namespace: str, key: str, value: str) -> str:
        """Save long-term memory under namespace/key."""
        return _invoke_schema_tool("save_memory", namespace=namespace, key=key, value=value)

    @lc_tool
    def task_create(subject: str, description: str = "", blocked_by: list[str] | None = None) -> str:
        """Create a persistent task."""
        return _invoke_schema_tool(
            "task_create",
            subject=subject,
            description=description,
            blocked_by=blocked_by,
        )

    @lc_tool
    def task_update(task_id: str, status: str | None = None, owner: str | None = None) -> str:
        """Update a persistent task."""
        return _invoke_schema_tool("task_update", task_id=task_id, status=status, owner=owner)

    @lc_tool
    def task_get(task_id: str) -> str:
        """Read a task."""
        return _invoke_schema_tool("task_get", task_id=task_id)

    @lc_tool
    def task_list() -> str:
        """List tasks."""
        return _invoke_schema_tool("task_list")

    @lc_tool
    def task_complete(task_id: str) -> str:
        """Complete a task and unblock dependents."""
        return _invoke_schema_tool("task_complete", task_id=task_id)

    @lc_tool
    def background_run(command: str, timeout: int = 3600) -> str:
        """Start a background command."""
        return _invoke_schema_tool("background_run", command=command, timeout=timeout)

    @lc_tool
    def background_check(runtime_task_id: str) -> str:
        """Check a background command."""
        return _invoke_schema_tool("background_check", runtime_task_id=runtime_task_id)

    @lc_tool
    def schedule_create(cron: str, prompt: str, recurring: bool = True, durable: bool = True) -> str:
        """Create a schedule that enqueues notifications when due."""
        return _invoke_schema_tool(
            "schedule_create",
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )

    @lc_tool
    def schedule_list() -> str:
        """List due schedules as notifications."""
        return _invoke_schema_tool("schedule_list")

    @lc_tool
    def schedule_delete(schedule_id: str) -> str:
        """Delete a schedule."""
        return _invoke_schema_tool("schedule_delete", schedule_id=schedule_id)

    @lc_tool
    def team_spawn(name: str, role: str = "worker", prompt: str = "") -> str:
        """Spawn a teammate/subagent record."""
        return _invoke_schema_tool("team_spawn", name=name, role=role, prompt=prompt)

    @lc_tool
    def send_message(teammate_id: str, content: str, request_id: str | None = None) -> str:
        """Send protocol message to a teammate inbox."""
        return _invoke_schema_tool(
            "send_message",
            teammate_id=teammate_id,
            content=content,
            request_id=request_id,
        )

    @lc_tool
    def request_shutdown(teammate_id: str, reason: str = "") -> str:
        """Request teammate shutdown."""
        return _invoke_schema_tool("request_shutdown", teammate_id=teammate_id, reason=reason)

    @lc_tool
    def submit_plan_approval(request_id: str, approved: bool, notes: str = "") -> str:
        """Submit a plan approval response."""
        return _invoke_schema_tool(
            "submit_plan_approval",
            request_id=request_id,
            approved=approved,
            notes=notes,
        )

    @lc_tool
    def claim_task(task_id: str, owner: str) -> str:
        """Claim a task with a lock."""
        return _invoke_schema_tool("claim_task", task_id=task_id, owner=owner)

    @lc_tool
    def worktree_create(task_id: str, base_path: str | None = None) -> str:
        """Create a worktree registry entry."""
        return _invoke_schema_tool("worktree_create", task_id=task_id, base_path=base_path)

    @lc_tool
    def worktree_enter(worktree_id: str) -> str:
        """Enter a registered worktree."""
        return _invoke_schema_tool("worktree_enter", worktree_id=worktree_id)

    @lc_tool
    def worktree_run(worktree_id: str, command: str, timeout: int = 60) -> str:
        """Run a command in a worktree."""
        return _invoke_schema_tool(
            "worktree_run",
            worktree_id=worktree_id,
            command=command,
            timeout=timeout,
        )

    @lc_tool
    def worktree_closeout(worktree_id: str, mode: str = "keep") -> str:
        """Close out a worktree, optionally removing it if clean."""
        return _invoke_schema_tool("worktree_closeout", worktree_id=worktree_id, mode=mode)

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


def _invoke_schema_tool(name: str, **kwargs: Any) -> str:
    """Execute a model-visible StructuredTool through the shared runtime."""
    call = {"id": "schema-call", "name": name, "args": kwargs}
    result = _runtime().execute([call], skip_permissions=True)[0]
    return result.content


def drain_notifications(state: AgentState) -> dict[str, Any]:
    if state.get("notifications"):
        return {"transition_reason": "notifications_drained"}
    return {"transition_reason": "no_notifications"}


def build_prompt(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    compacted = compact_check(state, config=cfg)
    if compacted.get("transition_reason") != "compact_not_needed":
        prompt_input = {**state, **compacted}
        prompt_input["prompt_state"] = invalidate_prompt_cache(prompt_input)
        compacted["system_prompt"] = _safe_build_system_prompt(prompt_input, cfg)
        compacted["prompt_state"] = prompt_input.get("prompt_state", {})
        return compacted
    update: dict[str, Any] = {"transition_reason": "prompt_built"}
    if not state.get("context_messages"):
        update["context_messages"] = list(state.get("messages", []))
    prompt_input = {**state, **update}
    update["system_prompt"] = _safe_build_system_prompt(prompt_input, cfg)
    update["prompt_state"] = prompt_input.get("prompt_state", {})
    return update


def _safe_build_system_prompt(state: AgentState | dict[str, Any], config: Config) -> str:
    try:
        return build_system_prompt(state, config)
    except Exception as exc:
        prompt_state = dict(state.get("prompt_state") or {})
        prompt_state["last_error"] = f"{type(exc).__name__}: {exc}"
        state["prompt_state"] = prompt_state
        return SYSTEM_PROMPT


def call_model(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    """Call the configured model unless pending tool calls already exist."""
    cfg = config or get_config()
    pending = state.get("pending_tool_calls") or state.get("tool_calls")
    if pending:
        return {
            "pending_tool_calls": pending,
            "tool_calls": pending,
            "transition_reason": "pending_tools_reused",
        }

    model_context = state.get("context_messages") or state.get("messages", [])
    system_prompt = state.get("system_prompt") or _safe_build_system_prompt(state, cfg)
    messages = [SystemMessage(content=system_prompt)] + model_context
    protocol_errors = validate_tool_message_protocol(messages)
    if protocol_errors:
        return {
            "error": "Invalid message protocol before model call: " + "; ".join(protocol_errors),
            "transition_reason": "message_protocol_error",
        }
    _sanitize_messages_for_utf8(messages)

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
            }

    _sanitize_message_for_utf8(response)
    next_context = list(model_context) + [response]
    if getattr(response, "tool_calls", None):
        return {
            "messages": [response],
            "context_messages": next_context,
            "pending_tool_calls": response.tool_calls,
            "tool_calls": response.tool_calls,
            "transition_reason": "model_requested_tools",
        }

    return {
        "messages": [response],
        "context_messages": next_context,
        "final_response": response.content,
        "transition_reason": "model_final_response",
    }


def route_model_response(state: AgentState) -> str:
    if state.get("transition_reason") in {"transient_model_retry", "context_compact_retry"}:
        return "retry"
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "tools"
    if state.get("error"):
        return "final"
    return "final" if state.get("final_response") else "final"


def permission_gate(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    mode = state.get("permission_mode") or (config or get_config()).permission_mode
    pending = list(state.get("pending_tool_calls") or state.get("tool_calls") or [])
    allowed: list[dict[str, Any]] = list(state.get("approved_tool_calls") or [])
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
                "approved_tool_calls": allowed,
                "pending_tool_calls": pending[index:],
                "tool_calls": pending[index:],
                "tool_results": denied,
                "transition_reason": "permission_interrupt_required",
            }
        allowed.append(tool_call)

    return {
        "pending_tool_calls": allowed,
        "tool_calls": allowed,
        "approved_tool_calls": [],
        "tool_results": denied,
        "pending_permission_request": None,
        "transition_reason": "permission_checked",
    }


def route_permission(state: AgentState) -> str:
    if state.get("pending_permission_request"):
        return "interrupt"
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "execute"
    if state.get("approved_tool_calls"):
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
    remaining = _remove_current_permission_call(remaining, request.get("tool_call_id"))
    if approved:
        approved_calls = list(state.get("approved_tool_calls") or []) + [call]
        return {
            "pending_permission_request": None,
            "approved_tool_calls": approved_calls,
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
        "pending_tool_calls": remaining,
        "tool_calls": remaining,
        "tool_results": list(state.get("tool_results") or []) + [denied],
        "transition_reason": "permission_denied",
    }


def route_after_human_permission(state: AgentState) -> str:
    if state.get("pending_tool_calls") or state.get("tool_calls"):
        return "permission"
    if state.get("approved_tool_calls"):
        return "execute"
    return "append"


def run_pre_tool_hooks(state: AgentState) -> dict[str, Any]:
    return {"transition_reason": "pre_tool_hooks_complete"}


def execute_tools(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    calls = (
        state.get("pending_tool_calls")
        or state.get("tool_calls")
        or state.get("approved_tool_calls")
        or []
    )
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
    merged = _merge_tool_results_in_call_order(
        state,
        [result.model_dump() for result in results],
    )
    return {
        "pending_tool_calls": [],
        "approved_tool_calls": [],
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
    compact_state = dict(state.get("compact_state") or {})
    for item in state.get("tool_results", []):
        envelope = item if isinstance(item, ToolResultEnvelope) else ToolResultEnvelope.model_validate(item)
        manual_request = _compact_request_from_envelope(envelope)
        if manual_request:
            compact_state["pending_manual_request"] = manual_request
        messages.append(
            ToolMessage(
                content=envelope.model_dump_json(),
                tool_call_id=envelope.tool_call_id or "unknown",
            )
        )
    return {
        "messages": messages,
        "context_messages": (
            list(state.get("context_messages") or state.get("messages", [])) + messages
        ),
        "compact_state": compact_state,
        "tool_results": [],
        "transition_reason": "tool_results_appended",
    }


def compact_check(
    state: AgentState,
    config: Config | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    if not messages:
        return {
            "context_messages": [],
            "transition_reason": "compact_not_needed",
        }

    manual_summary = _pending_manual_compact_summary(state)
    context_hash = _context_fingerprint(messages, manual_summary)
    compact_state = dict(state.get("compact_state") or {})
    if (
        not force
        and not manual_summary
        and state.get("context_messages")
        and compact_state.get("last_context_hash") == context_hash
    ):
        return {
            "compact_state": compact_state,
            "transition_reason": "compact_not_needed",
        }

    policy = get_compaction_policy(config or get_config())
    compacted = compact_messages(
        messages,
        policy,
        turn_count=state.get("turn_count", 0),
        manual_summary=manual_summary,
        force_micro=_should_time_based_microcompact(state, config or get_config()),
    )
    compacted = _add_pre_compact_context(compacted, state, config or get_config())
    compacted = _maybe_add_model_compact_summary(compacted, config or get_config(), state)
    compacted = _add_post_compact_context(compacted, state, config or get_config())

    compact_state = _updated_compact_state(state, compacted, context_hash)
    if compacted.mode == "summary":
        transition_reason = "summary_compact_complete"
    elif compacted.mode == "micro":
        transition_reason = "micro_compact_complete"
    else:
        transition_reason = "compact_not_needed"
    return {
        "context_messages": compacted.context_messages,
        "compact_state": compact_state,
        "transition_reason": transition_reason,
    }


def recovery_handler(state: AgentState, config: Config | None = None) -> dict[str, Any]:
    error = state.get("error")
    if error:
        if _is_context_too_long_error(str(error)) and _recovery_budget(state, "context_retry_budget") > 0:
            compacted = compact_check(state, config=config, force=True)
            recovery = _consume_recovery_budget(state, "context_retry_budget")
            update = {
                "error": None,
                "transition_reason": "context_compact_retry",
                "recovery_state": recovery,
            }
            update.update(compacted)
            update["transition_reason"] = "context_compact_retry"
            return update
        if _is_transient_model_error(str(error)) and _recovery_budget(state, "transient_retry_budget") > 0:
            time.sleep(_transient_retry_delay(state))
            return {
                "error": None,
                "transition_reason": "transient_model_retry",
                "recovery_state": _consume_recovery_budget(state, "transient_retry_budget"),
            }
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
    cfg = get_config()
    runtime = ToolExecutionRuntime(cfg.working_path, config=cfg, output_limit=cfg.output_limit)
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


def _recovery_budget(state: AgentState, key: str) -> int:
    return int((state.get("recovery_state") or {}).get(key, 0))


def _is_transient_model_error(error: str) -> bool:
    normalized = error.lower()
    markers = (
        "429",
        "rate limit",
        "overloaded",
        "temporarily",
        "timeout",
        "timed out",
        "connection",
        "network",
        "server error",
        "503",
        "502",
    )
    return any(marker in normalized for marker in markers)


def _is_context_too_long_error(error: str) -> bool:
    normalized = error.lower()
    markers = (
        "context length",
        "context_length",
        "context too long",
        "prompt too long",
        "maximum context",
        "token limit",
        "too many tokens",
        "413",
    )
    return any(marker in normalized for marker in markers)


def _transient_retry_delay(state: AgentState) -> float:
    remaining = _recovery_budget(state, "transient_retry_budget")
    return min(2.0, 0.5 * (3 - remaining))


def _should_time_based_microcompact(state: AgentState, config: Config) -> bool:
    gap = int(getattr(config, "time_based_microcompact_turn_gap", 0) or 0)
    if gap <= 0:
        return False
    compact_state = state.get("compact_state") or {}
    last_turn = int(compact_state.get("last_main_loop_assistant_turn", 0) or 0)
    return int(state.get("turn_count", 0) or 0) - last_turn >= gap


def _remove_current_permission_call(
    pending: list[dict[str, Any]],
    tool_call_id: str | None,
) -> list[dict[str, Any]]:
    if not pending:
        return []
    if not tool_call_id:
        return pending[1:]
    for index, call in enumerate(pending):
        if call.get("id") == tool_call_id:
            return pending[:index] + pending[index + 1 :]
    return pending[1:]


def _merge_tool_results_in_call_order(
    state: AgentState,
    new_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = list(state.get("tool_results") or []) + new_results
    order = _last_tool_call_order(state)
    if not order:
        return results
    ordered_items = sorted(
        enumerate(results),
        key=lambda item: (order.get(item[1].get("tool_call_id"), len(order)), item[0]),
    )
    return [result for _index, result in ordered_items]


def _last_tool_call_order(state: AgentState) -> dict[str, int]:
    last_ai = next((m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), None)
    tool_calls = getattr(last_ai, "tool_calls", None) or []
    return {
        call.get("id"): index
        for index, call in enumerate(tool_calls)
        if isinstance(call, dict) and call.get("id")
    }


def _maybe_add_model_compact_summary(
    compacted: CompactionOutput,
    config: Config,
    state: AgentState,
) -> CompactionOutput:
    if compacted.mode != "summary" or not compacted.summary:
        return compacted
    if config.llm_model == "mock" or not getattr(config, "compact_use_model_summary", True):
        return compacted
    if not config.llm_api_key:
        compacted.token_budget["model_summary_skipped"] = "missing_api_key"
        return compacted
    if int((state.get("compact_state") or {}).get("consecutive_failures", 0)) >= int(
        getattr(config, "compact_failure_circuit_breaker", 3)
    ):
        compacted.token_budget["model_summary_skipped"] = "circuit_breaker"
        return compacted

    model_summary, failed = _generate_model_compact_summary(compacted.summary, config)
    if failed:
        compacted.token_budget["model_summary_failed"] = True
    if not model_summary:
        return compacted

    summary = dict(compacted.summary)
    summary["model_summary"] = model_summary
    context_messages = list(compacted.context_messages)
    if len(context_messages) >= 2:
        context_messages[1] = HumanMessage(content=format_summary(summary))
    return CompactionOutput(
        mode=compacted.mode,
        context_messages=context_messages,
        summary=summary,
        boundary_id=compacted.boundary_id,
        token_budget=compacted.token_budget,
        micro_compacted_tool_results=compacted.micro_compacted_tool_results,
    )


def _generate_model_compact_summary(summary: dict[str, Any], config: Config) -> tuple[str | None, bool]:
    llm = get_llm(config=config)
    prompts = [build_model_compact_prompt(format_summary(summary))]
    if int(getattr(config, "compact_summary_retry_budget", 1)) > 0:
        prompts.append(build_model_compact_prompt(format_summary(summary), short=True))

    failed = False
    for prompt in prompts:
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a context compaction summarizer. You have no tools. "
                            "If you try to call tools, the compaction turn is wasted."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            failed = True
            if _is_context_too_long_error(str(exc)):
                continue
            return None, True
        content = format_model_compact_summary(getattr(response, "content", ""))
        if content:
            return content, failed
        failed = True
    return None, failed


def _add_pre_compact_context(
    compacted: CompactionOutput,
    state: AgentState,
    config: Config,
) -> CompactionOutput:
    if compacted.mode != "summary" or not compacted.summary or not compacted.boundary_id:
        return compacted
    summary = dict(compacted.summary)
    transcript_path = write_transcript(
        list(state.get("messages", [])),
        config.working_path,
        compacted.boundary_id,
    )
    summary["transcript_path"] = transcript_path
    pre_hooks = run_compact_hook(config.working_path, "pre_compact")
    if pre_hooks:
        summary["pre_compact_hooks"] = pre_hooks
    context_messages = list(compacted.context_messages)
    if len(context_messages) >= 2:
        context_messages[1] = HumanMessage(content=format_summary(summary))
    return CompactionOutput(
        mode=compacted.mode,
        context_messages=context_messages,
        summary=summary,
        boundary_id=compacted.boundary_id,
        token_budget=compacted.token_budget,
        micro_compacted_tool_results=compacted.micro_compacted_tool_results,
    )


def _add_post_compact_context(
    compacted: CompactionOutput,
    state: AgentState,
    config: Config,
) -> CompactionOutput:
    if compacted.mode != "summary" or not compacted.summary:
        return compacted
    post_hooks = run_compact_hook(config.working_path, "post_compact")
    rehydration_text = build_rehydration_text(
        state,
        transcript_path=compacted.summary.get("transcript_path"),
        post_compact_hooks=post_hooks,
    )
    if rehydration_text.strip() == "Runtime context after compaction:":
        return compacted
    summary = dict(compacted.summary)
    if post_hooks:
        summary["post_compact_hooks"] = post_hooks
    context_messages = list(compacted.context_messages)
    insert_at = _trailing_tool_group_start(context_messages)
    context_messages.insert(insert_at, HumanMessage(content=rehydration_text))
    return CompactionOutput(
        mode=compacted.mode,
        context_messages=context_messages,
        summary=summary,
        boundary_id=compacted.boundary_id,
        token_budget=compacted.token_budget,
        micro_compacted_tool_results=compacted.micro_compacted_tool_results,
    )


def _trailing_tool_group_start(messages: list[Any]) -> int:
    if not messages or not isinstance(messages[-1], ToolMessage):
        return len(messages)
    index = len(messages) - 1
    while index >= 0 and isinstance(messages[index], ToolMessage):
        index -= 1
    if index >= 0 and getattr(messages[index], "tool_calls", None):
        return index
    return len(messages)


def _updated_compact_state(
    state: AgentState,
    compacted: Any,
    context_hash: str | None = None,
) -> dict[str, Any]:
    compact_state = dict(state.get("compact_state") or {})
    summaries = list(compact_state.get("summaries") or [])
    history = list(compact_state.get("compaction_history") or [])
    if compacted.summary:
        summaries.append(compacted.summary)
        compact_state["last_boundary_id"] = compacted.boundary_id
        compact_state["transcript_path"] = compacted.summary.get("transcript_path")
        compact_state["recent_messages_kept"] = len(compacted.context_messages)
        compact_state["pending_manual_request"] = None
    if compacted.mode in {"summary", "micro"}:
        compact_state["last_compacted_turn"] = state.get("turn_count", 0)
        compact_state["last_main_loop_assistant_turn"] = state.get("turn_count", 0)
    if context_hash:
        compact_state["last_context_hash"] = context_hash
    compact_state["mode"] = compacted.mode
    compact_state["summaries"] = summaries
    compact_state["token_budget"] = compacted.token_budget
    compact_state["warning_state"] = _compact_warning_state(compacted.token_budget)
    compact_state["micro_compacted_tool_results"] = compacted.micro_compacted_tool_results
    if compacted.token_budget.get("model_summary_failed"):
        compact_state["consecutive_failures"] = int(compact_state.get("consecutive_failures", 0)) + 1
    elif compacted.token_budget.get("model_summary_skipped") == "circuit_breaker":
        compact_state["consecutive_failures"] = int(compact_state.get("consecutive_failures", 0))
    elif compacted.summary and compacted.summary.get("model_summary"):
        compact_state["consecutive_failures"] = 0
    history.append(
        {
            "mode": compacted.mode,
            "turn_count": state.get("turn_count", 0),
            "boundary_id": compacted.boundary_id,
            "estimated_tokens": compacted.token_budget.get("estimated_tokens"),
            "after_micro_tokens": compacted.token_budget.get("after_micro_tokens"),
            "after_summary_tokens": compacted.token_budget.get("after_summary_tokens"),
        }
    )
    compact_state["compaction_history"] = history[-20:]
    return compact_state


def _compact_warning_state(token_budget: dict[str, Any]) -> dict[str, Any]:
    estimated = int(token_budget.get("estimated_tokens") or 0)
    warning_threshold = int(token_budget.get("warning_threshold") or 0)
    auto_threshold = int(token_budget.get("auto_compact_threshold") or 0)
    context_window = int(token_budget.get("context_window_tokens") or 0)
    return {
        "estimated_tokens": estimated,
        "warning_threshold": warning_threshold,
        "auto_compact_threshold": auto_threshold,
        "context_window_tokens": context_window,
        "warning": bool(warning_threshold and estimated >= warning_threshold),
        "auto_compact": bool(auto_threshold and estimated >= auto_threshold),
    }


def _pending_manual_compact_summary(state: AgentState) -> str | None:
    request = (state.get("compact_state") or {}).get("pending_manual_request")
    if isinstance(request, dict):
        return request.get("summary") or ""
    for item in state.get("tool_results", []) or []:
        envelope = (
            item
            if isinstance(item, ToolResultEnvelope)
            else ToolResultEnvelope.model_validate(item)
        )
        if envelope.metadata.get("tool_name") != "compact":
            continue
        try:
            import json

            payload = json.loads(envelope.content)
        except Exception:
            return envelope.content
        if payload.get("mode") == "manual":
            return payload.get("summary") or ""
    return None


def _compact_request_from_envelope(envelope: ToolResultEnvelope) -> dict[str, Any] | None:
    if envelope.metadata.get("tool_name") != "compact":
        return None
    try:
        payload = json.loads(envelope.content)
    except Exception:
        payload = {"mode": "manual", "summary": envelope.content}
    if payload.get("mode") != "manual":
        return None
    return {"mode": "manual", "summary": payload.get("summary") or ""}


def _context_fingerprint(messages: list[Any], manual_summary: str | None = None) -> str:
    parts: list[dict[str, Any]] = []
    for message in messages:
        parts.append(
            {
                "type": getattr(message, "type", type(message).__name__),
                "content": getattr(message, "content", ""),
                "tool_calls": getattr(message, "tool_calls", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
        )
    payload = json.dumps(
        {"messages": parts, "manual_summary": manual_summary},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summarize_messages(messages: list[Any]) -> dict[str, Any]:
    policy = get_compaction_policy(get_config())
    compacted = compact_messages(messages, policy, turn_count=0, manual_summary="")
    if compacted.summary:
        return compacted.summary
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
