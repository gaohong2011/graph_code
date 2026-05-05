"""Message protocol validation for OpenAI-compatible chat models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def validate_tool_message_protocol(messages: Sequence[BaseMessage]) -> list[str]:
    """Return protocol errors for assistant tool calls missing tool results.

    OpenAI-compatible chat APIs require every assistant message containing
    tool_calls to be followed immediately by one ToolMessage per tool_call_id
    before another human/assistant/system message appears.
    """
    errors: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            expected_ids = [_tool_call_id(call) for call in tool_calls]
            missing_ids = [tool_call_id for tool_call_id in expected_ids if not tool_call_id]
            if missing_ids:
                errors.append(f"assistant message at index {index} has tool_call without id")

            expected_set = {tool_call_id for tool_call_id in expected_ids if tool_call_id}
            tool_window = messages[index + 1 : index + 1 + len(expected_ids)]
            if len(tool_window) < len(expected_ids):
                absent = sorted(expected_set)
                errors.append(
                    f"assistant message at index {index} is missing tool results for {absent}"
                )
                break

            actual_ids: list[str] = []
            for offset, candidate in enumerate(tool_window, start=1):
                if not isinstance(candidate, ToolMessage):
                    errors.append(
                        f"assistant tool_call ids {sorted(expected_set)} must be followed by "
                        f"ToolMessage entries; found {candidate.type!r} at index {index + offset}"
                    )
                    break
                actual_ids.append(candidate.tool_call_id)

            actual_set = set(actual_ids)
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            if missing:
                errors.append(
                    f"assistant message at index {index} is missing tool results for {missing}"
                )
            if extra:
                errors.append(
                    f"tool messages after assistant index {index} reference unknown ids {extra}"
                )
            index += 1 + len(expected_ids)
            continue

        if isinstance(message, ToolMessage):
            errors.append(
                f"ToolMessage at index {index} with id {message.tool_call_id!r} has no "
                "immediately preceding assistant tool_call"
            )
        index += 1
    return errors


def _tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)
