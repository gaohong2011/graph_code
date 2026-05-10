"""Build the model-facing system prompt."""

from __future__ import annotations

from typing import Any

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
        project_instruction_section(config),
        memory_section(config),
        environment_section(config),
    ]
    prompt_state = dict(state.get("prompt_state") or {})
    prompt_state["invalidated"] = False
    prompt_state["last_error"] = None
    state["prompt_state"] = prompt_state
    return "\n\n".join(section for section in sections if section)
