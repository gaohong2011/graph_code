"""Token budgeting and compaction policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class CompactionPolicy:
    """Configuration for deciding when and how to compact context."""

    context_window_tokens: int
    auto_compact_ratio: float
    micro_compact_ratio: float
    recent_messages: int
    keep_tool_results: int
    summary_max_chars: int = 12000
    tool_result_preview_chars: int = 240
    min_tool_result_tokens: int = 128
    message_count_threshold: int = 40

    @property
    def auto_compact_threshold(self) -> int:
        return max(1, int(self.context_window_tokens * self.auto_compact_ratio))

    @property
    def micro_compact_threshold(self) -> int:
        return max(1, int(self.context_window_tokens * self.micro_compact_ratio))


def get_compaction_policy(config: Any) -> CompactionPolicy:
    """Build a policy from Config while keeping defaults production-oriented."""

    return CompactionPolicy(
        context_window_tokens=int(getattr(config, "context_window_tokens", 200_000)),
        auto_compact_ratio=float(getattr(config, "auto_compact_ratio", 0.82)),
        micro_compact_ratio=float(getattr(config, "micro_compact_ratio", 0.68)),
        recent_messages=int(getattr(config, "compact_recent_messages", 12)),
        keep_tool_results=int(getattr(config, "micro_compact_keep_tool_results", 4)),
        summary_max_chars=int(getattr(config, "compact_summary_max_chars", 12_000)),
        tool_result_preview_chars=int(getattr(config, "micro_compact_preview_chars", 240)),
        min_tool_result_tokens=int(getattr(config, "micro_compact_min_tool_result_tokens", 128)),
        message_count_threshold=int(getattr(config, "compact_message_count_threshold", 40)),
    )


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate tokens conservatively without provider-specific tokenizers."""

    content_tokens = estimate_text_tokens(_content_to_text(message.content))
    tool_tokens = 0
    for tool_call in getattr(message, "tool_calls", None) or []:
        tool_tokens += estimate_text_tokens(str(tool_call))
    return content_tokens + tool_tokens + 8


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 1) // 2)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        return " ".join(f"{key}: {_content_to_text(value)}" for key, value in content.items())
    return str(content)
