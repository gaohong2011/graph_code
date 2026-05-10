"""Compatibility support for the legacy save_memory tool."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from .paths import memory_paths_for_project
from .types import normalize_memory_type


def save_legacy_memory(config: Any, namespace: str, key: str, value: str) -> str:
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True, exist_ok=True)
    if not paths.memory_index.exists():
        paths.memory_index.write_text("", encoding="utf-8")

    memory_type = normalize_memory_type(namespace) or "reference"
    slug = _slug(f"{memory_type} {key}")
    topic = paths.memory_dir / f"{slug}.md"
    title = key.strip() or "memory"
    content = "\n".join(
        [
            "---",
            f"name: {title}",
            f"description: {value.strip()[:160]}",
            f"type: {memory_type}",
            f"updated_at: {date.today().isoformat()}",
            "---",
            "",
            value.strip(),
            "",
        ]
    )
    topic.write_text(content, encoding="utf-8")

    entry = f"- [{title}]({topic.name}) - {value.strip()[:120]}"
    index = paths.memory_index.read_text(encoding="utf-8", errors="ignore")
    if topic.name not in index:
        suffix = "\n" if index and not index.endswith("\n") else ""
        paths.memory_index.write_text(index + suffix + entry + "\n", encoding="utf-8")

    return json.dumps({"path": topic.as_posix(), "type": memory_type}, ensure_ascii=False)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "memory"
