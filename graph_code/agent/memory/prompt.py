"""Prompt text and index loading for global memory."""

from __future__ import annotations

from typing import Any

from .paths import memory_paths_for_project

MAX_INDEX_LINES = 200
MAX_INDEX_CHARS = 25000


def build_memory_prompt(config: Any) -> str | None:
    if getattr(config, "memory_disabled", False):
        return None
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True, exist_ok=True)
    if not paths.memory_index.exists():
        paths.memory_index.write_text("", encoding="utf-8")
    return "\n".join(
        [
            "# Memory",
            "",
            f"You have a persistent, file-based memory system at `{paths.memory_dir}`.",
            "`MEMORY.md` is an index. Store each durable memory in its own markdown topic file.",
            "",
            "Use this frontmatter shape for topic files:",
            "```yaml",
            "---",
            "name: short descriptive name",
            "description: one-line recall hook",
            "type: feedback",
            "updated_at: 2026-05-06",
            "---",
            "```",
            "",
            "Types: `user`, `feedback`, `project`, `reference`.",
            "",
            "What not to save:",
            "- Code structure, conventions, architecture, or file paths that can be read from the repository.",
            "- Git history or recent file changes.",
            "- Secrets, credentials, API keys, tokens, or sensitive personal data.",
            "- Temporary task state that only matters in the current conversation.",
            "- Information already documented in project instruction files.",
            "",
            "When saving memory, update an existing topic if one already covers the subject.",
            "When forgetting memory, remove or edit the relevant topic and update `MEMORY.md`.",
        ]
    )


def load_memory_index_context(config: Any) -> str:
    if getattr(config, "memory_disabled", False):
        return ""
    paths = memory_paths_for_project(config)
    try:
        raw = paths.memory_index.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""
    if not raw:
        return "MEMORY.md is currently empty."
    lines = raw.splitlines()
    clipped = "\n".join(lines[:MAX_INDEX_LINES])
    if len(clipped) > MAX_INDEX_CHARS:
        clipped = clipped[:MAX_INDEX_CHARS]
    if len(lines) > MAX_INDEX_LINES or len(raw) > len(clipped):
        clipped += "\n\n> Memory index truncated. Keep entries short and move detail into topic files."
    return f"Contents of MEMORY.md:\n\n{clipped}"
