"""Session memory threshold helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..compaction.policy import estimate_messages_tokens


def should_update_session_memory(state: dict[str, Any], config: Any) -> bool:
    if not getattr(config, "session_memory_enabled", False):
        return False
    if state.get("pending_tool_calls") or state.get("tool_calls") or state.get("pending_permission_request"):
        return False
    last = state.get("messages", [])[-1:] or []
    if last and isinstance(last[0], AIMessage) and getattr(last[0], "tool_calls", None):
        return False
    current_tokens = estimate_messages_tokens(list(state.get("messages", [])))
    session_state = state.get("session_memory_state") or {}
    if not session_state.get("initialized"):
        return current_tokens >= int(getattr(config, "session_memory_init_tokens", 10000))
    growth = current_tokens - int(session_state.get("tokens_at_last_update", 0))
    return growth >= int(getattr(config, "session_memory_update_tokens", 5000))
