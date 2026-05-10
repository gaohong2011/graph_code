"""Use session memory as a compact summary source."""

from __future__ import annotations

from typing import Any

from ..memory.paths import memory_paths_for_project
from .prompt import DEFAULT_SESSION_MEMORY_TEMPLATE

MAX_SESSION_MEMORY_CHARS = 24000


def load_session_memory_for_compact(config: Any) -> str | None:
    if not getattr(config, "session_memory_enabled", False):
        return None
    path = memory_paths_for_project(config).session_memory_file
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not content or content == DEFAULT_SESSION_MEMORY_TEMPLATE.strip():
        return None
    if len(content) > MAX_SESSION_MEMORY_CHARS:
        return content[:MAX_SESSION_MEMORY_CHARS] + "\n\n[Session memory truncated]"
    return content
