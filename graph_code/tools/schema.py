"""Shared tool schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResultEnvelope(BaseModel):
    """Uniform result returned by every tool implementation."""

    ok: bool
    content: str
    is_error: bool = False
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = "unknown"

    @classmethod
    def success(
        cls,
        content: str,
        tool_call_id: str = "unknown",
        metadata: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> "ToolResultEnvelope":
        return cls(
            ok=True,
            content=content,
            is_error=False,
            attachments=attachments or [],
            metadata=metadata or {},
            tool_call_id=tool_call_id,
        )

    @classmethod
    def error(
        cls,
        content: str,
        tool_call_id: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResultEnvelope":
        return cls(
            ok=False,
            content=content,
            is_error=True,
            metadata=metadata or {},
            tool_call_id=tool_call_id,
        )
