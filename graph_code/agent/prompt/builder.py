"""Build the model-facing system prompt."""

from __future__ import annotations

from typing import Any

from ..memory.relevance import build_relevant_memory_context, select_relevant_memories
from .cache import cached_section
from .sections import (
    context_behavior_section,
    environment_section,
    identity_section,
    memory_section,
    project_instruction_section,
    task_behavior_section,
    tool_behavior_section,
)


def build_system_prompt(state: dict[str, Any], config: Any) -> str:
    sections = [
        cached_section(state, "identity", identity_section),
        cached_section(state, "task_behavior", task_behavior_section),
        cached_section(state, "tool_behavior", tool_behavior_section),
        cached_section(state, "context_behavior", context_behavior_section),
        project_instruction_section(config, active_paths=_active_file_paths(state)),
        memory_section(config),
        environment_section(config),
    ]
    latest_query = _latest_user_text(state)
    relevant = select_relevant_memories(latest_query, config) if latest_query else []
    relevant_context = build_relevant_memory_context(relevant)
    if relevant_context:
        sections.append(relevant_context)
        memory_state = dict(state.get("memory_state") or {})
        memory_state["surfaced_memories"] = [path.as_posix() for path in relevant]
        state["memory_state"] = memory_state
    prompt_state = dict(state.get("prompt_state") or {})
    prompt_state["invalidated"] = False
    prompt_state["last_error"] = None
    state["prompt_state"] = prompt_state
    return "\n\n".join(section for section in sections if section)


def _latest_user_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if getattr(message, "type", "") == "human":
            return str(getattr(message, "content", ""))
    return ""


def _active_file_paths(state: dict[str, Any]) -> list[str]:
    file_state = state.get("file_context_state") or {}
    recent = file_state.get("recent_files") or []
    paths: list[str] = []
    for item in recent:
        if not isinstance(item, dict):
            continue
        value = item.get("path")
        if isinstance(value, str) and value:
            paths.append(value)
    return paths
