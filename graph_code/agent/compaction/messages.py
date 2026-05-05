"""Protocol-safe message compaction."""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from .policy import (
    CompactionPolicy,
    estimate_message_tokens,
    estimate_messages_tokens,
)

COMPACTABLE_TOOL_NAMES = {
    "read_file",
    "search_files",
    "bash",
    "worktree_run",
    "_read_file",
    "_grep_search",
    "_glob_search",
    "_bash_command",
    "bash_command",
    "grep_search",
    "glob_search",
}


@dataclass(frozen=True)
class CompactionOutput:
    """Result of applying compaction to model context."""

    mode: str
    context_messages: list[BaseMessage]
    summary: dict[str, Any] | None
    boundary_id: str | None
    token_budget: dict[str, Any]
    micro_compacted_tool_results: int = 0


@dataclass(frozen=True)
class _MessageGroup:
    start: int
    end: int
    messages: list[BaseMessage]


def compact_messages(
    messages: list[BaseMessage],
    policy: CompactionPolicy,
    *,
    turn_count: int = 0,
    manual_summary: str | None = None,
    force_micro: bool = False,
) -> CompactionOutput:
    """Build the model-visible context for the next model call."""

    original_tokens = estimate_messages_tokens(messages)
    token_budget = {
        "estimated_tokens": original_tokens,
        "context_window_tokens": policy.context_window_tokens,
        "micro_compact_threshold": policy.micro_compact_threshold,
        "auto_compact_threshold": policy.auto_compact_threshold,
        "warning_threshold": policy.warning_threshold,
    }

    micro_messages, compacted_count = micro_compact_tool_results(
        messages,
        policy,
        force=force_micro,
    )
    micro_tokens = estimate_messages_tokens(micro_messages)
    token_budget["after_micro_tokens"] = micro_tokens

    if (
        manual_summary
        or micro_tokens >= policy.auto_compact_threshold
        or len(messages) >= policy.message_count_threshold
    ):
        prefix, suffix = split_recent_protocol_suffix(micro_messages, policy.recent_messages)
        boundary_id = f"compact-{uuid.uuid4().hex[:12]}"
        summary = build_summary(
            prefix=prefix,
            suffix=suffix,
            all_messages=messages,
            boundary_id=boundary_id,
            turn_count=turn_count,
            manual_summary=manual_summary,
            max_chars=policy.summary_max_chars,
        )
        context_messages = [
            HumanMessage(content=f"[Context compacted: {boundary_id}]"),
            HumanMessage(content=format_summary(summary)),
            *suffix,
        ]
        token_budget["after_summary_tokens"] = estimate_messages_tokens(context_messages)
        return CompactionOutput(
            mode="summary",
            context_messages=context_messages,
            summary=summary,
            boundary_id=boundary_id,
            token_budget=token_budget,
            micro_compacted_tool_results=compacted_count,
        )

    if compacted_count and (original_tokens >= policy.micro_compact_threshold or force_micro):
        return CompactionOutput(
            mode="micro",
            context_messages=micro_messages,
            summary=None,
            boundary_id=None,
            token_budget=token_budget,
            micro_compacted_tool_results=compacted_count,
        )

    return CompactionOutput(
        mode="none",
        context_messages=list(messages),
        summary=None,
        boundary_id=None,
        token_budget=token_budget,
        micro_compacted_tool_results=0,
    )


def micro_compact_tool_results(
    messages: list[BaseMessage],
    policy: CompactionPolicy,
    *,
    force: bool = False,
) -> tuple[list[BaseMessage], int]:
    """Replace old bulky ToolMessage content with a compact marker."""

    tool_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
    ]
    tool_names = _tool_names_by_id(messages)
    keep_indexes = set(tool_indexes[-policy.keep_tool_results :])
    compacted = 0
    output: list[BaseMessage] = []

    for index, message in enumerate(messages):
        cloned = copy.deepcopy(message)
        if (
            isinstance(cloned, ToolMessage)
            and index not in keep_indexes
            and _is_compactable_tool(tool_names.get(cloned.tool_call_id))
            and (force or estimate_message_tokens(cloned) >= policy.min_tool_result_tokens)
        ):
            cloned.content = _compact_tool_content(str(cloned.content), policy)
            compacted += 1
        output.append(cloned)

    return output, compacted


def _tool_names_by_id(messages: list[BaseMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("id"):
                names[call["id"]] = str(call.get("name", ""))
    return names


def _is_compactable_tool(name: str | None) -> bool:
    if not name:
        return False
    return name in COMPACTABLE_TOOL_NAMES or name.startswith("mcp__")


def split_recent_protocol_suffix(
    messages: list[BaseMessage],
    recent_messages: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split messages into old prefix and recent suffix without cutting tool groups."""

    if not messages:
        return [], []
    groups = _group_protocol_units(messages)
    suffix_start_group = max(0, len(groups) - 1)
    kept_count = 0
    for group_index in range(len(groups) - 1, -1, -1):
        group = groups[group_index]
        kept_count += len(group.messages)
        suffix_start_group = group_index
        if kept_count >= recent_messages:
            break
    suffix_start = groups[suffix_start_group].start
    return list(messages[:suffix_start]), list(messages[suffix_start:])


def build_summary(
    *,
    prefix: list[BaseMessage],
    suffix: list[BaseMessage],
    all_messages: list[BaseMessage],
    boundary_id: str,
    turn_count: int,
    manual_summary: str | None,
    max_chars: int,
) -> dict[str, Any]:
    """Create an extractive compact summary with stable required fields."""

    last_human = next(
        (message for message in reversed(all_messages) if isinstance(message, HumanMessage)),
        None,
    )
    current_goal = (
        _preview(str(last_human.content), 240)
        if last_human
        else "Continue the coding task"
    )
    completed_actions = [_message_action(message) for message in prefix[-12:]]
    key_files = _extract_file_references(prefix)
    user_messages = [
        _preview(str(message.content), 300)
        for message in prefix
        if isinstance(message, HumanMessage)
    ][-8:]
    current_work = [_message_action(message) for message in suffix[-6:]]

    summary = {
        "boundary_id": boundary_id,
        "turn_count": turn_count,
        "current_goal": current_goal,
        "primary_request": current_goal,
        "completed_actions": completed_actions,
        "key_files": key_files,
        "key_decisions": _extract_decisions(prefix),
        "errors_and_fixes": _extract_errors(prefix),
        "user_messages": user_messages,
        "pending_tasks": [],
        "current_work": current_work,
        "next_step": current_goal,
    }
    if manual_summary:
        summary["manual_summary"] = _preview(manual_summary, max_chars)
    return _trim_summary(summary, max_chars)


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Context compacted. Continue from this summary and the recent verbatim messages below.",
        f"Boundary: {summary.get('boundary_id')}",
        f"Current goal: {summary.get('current_goal')}",
        f"Primary request: {summary.get('primary_request')}",
    ]
    if summary.get("model_summary"):
        lines.append("Model summary:")
        lines.append(str(summary["model_summary"]))
    if summary.get("transcript_path"):
        lines.append(f"Full transcript: {summary['transcript_path']}")
    if summary.get("pre_compact_hooks"):
        lines.append("PreCompact hooks:")
        for hook in summary["pre_compact_hooks"]:
            lines.append(f"- {hook.get('hook')}: {hook.get('content', '')}")
    lines.append("Completed actions:")
    lines.extend(f"- {item}" for item in summary.get("completed_actions", []) or ["None recorded"])
    lines.append("Key files:")
    lines.extend(f"- {item}" for item in summary.get("key_files", []) or ["None recorded"])
    lines.append("Key decisions:")
    lines.extend(f"- {item}" for item in summary.get("key_decisions", []) or ["None recorded"])
    lines.append("Errors and fixes:")
    lines.extend(f"- {item}" for item in summary.get("errors_and_fixes", []) or ["None recorded"])
    lines.append("User messages:")
    lines.extend(f"- {item}" for item in summary.get("user_messages", []) or ["None recorded"])
    lines.append("Current work:")
    lines.extend(f"- {item}" for item in summary.get("current_work", []) or ["None recorded"])
    if summary.get("manual_summary"):
        lines.append(f"Manual summary: {summary['manual_summary']}")
    lines.append(f"Next step: {summary.get('next_step')}")
    return "\n".join(lines)


def _group_protocol_units(messages: list[BaseMessage]) -> list[_MessageGroup]:
    groups: list[_MessageGroup] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            end = min(len(messages), index + 1 + len(tool_calls))
            groups.append(_MessageGroup(index, end, list(messages[index:end])))
            index = end
            continue
        groups.append(_MessageGroup(index, index + 1, [message]))
        index += 1
    return groups


def _compact_tool_content(content: str, policy: CompactionPolicy) -> str:
    persisted = _persisted_marker(content)
    preview = _preview(content, policy.tool_result_preview_chars)
    lines = ["[old tool result compacted]", f"Preview: {preview}"]
    if persisted:
        lines.append(persisted)
    return "\n".join(lines)


def _message_action(message: BaseMessage) -> str:
    if isinstance(message, ToolMessage):
        return f"tool_result {message.tool_call_id}: {_preview(str(message.content), 160)}"
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        names = ", ".join(
            str(call.get("name", "unknown"))
            for call in tool_calls
            if isinstance(call, dict)
        )
        return f"assistant requested tools: {names or 'unknown'}"
    return f"{message.type}: {_preview(str(message.content), 180)}"


def _extract_file_references(messages: list[BaseMessage]) -> list[str]:
    files: list[str] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if isinstance(tool_call, dict):
                args = tool_call.get("args") or {}
                for key in ("file_path", "path"):
                    value = args.get(key)
                    if isinstance(value, str):
                        files.append(value)
        for match in re.findall(
            r"[\w./-]+\.(?:py|md|txt|json|toml|yaml|yml|ts|tsx|js|jsx)",
            str(message.content),
        ):
            files.append(match)
    return _unique(files)[:12]


def _extract_decisions(messages: list[BaseMessage]) -> list[str]:
    decisions = []
    markers = ("decided", "decision", "选择", "决定", "采用")
    for message in messages:
        text = str(message.content)
        if any(marker in text.lower() for marker in markers):
            decisions.append(_preview(text, 220))
    return decisions[-6:]


def _extract_errors(messages: list[BaseMessage]) -> list[str]:
    errors = []
    markers = ("error", "failed", "traceback", "exception", "报错", "失败")
    for message in messages:
        text = str(message.content)
        if any(marker in text.lower() for marker in markers):
            errors.append(_preview(text, 220))
    return errors[-6:]


def _trim_summary(summary: dict[str, Any], max_chars: int) -> dict[str, Any]:
    rendered = str(summary)
    if len(rendered) <= max_chars:
        return summary
    trimmed = dict(summary)
    for key in ("completed_actions", "user_messages", "current_work"):
        values = list(trimmed.get(key) or [])
        while values and len(str(trimmed)) > max_chars:
            values.pop(0)
            trimmed[key] = values
    return trimmed


def _preview(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _persisted_marker(content: str) -> str | None:
    match = re.search(r"\[persisted-output: [^\]]+\]", content)
    return match.group(0) if match else None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
