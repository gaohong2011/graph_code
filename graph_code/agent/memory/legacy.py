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
    metadata_title = _frontmatter_scalar(title)
    metadata_description = _frontmatter_scalar(value, limit=160)
    content = "\n".join(
        [
            "---",
            f"name: {metadata_title}",
            f"description: {metadata_description}",
            f"type: {memory_type}",
            f"updated_at: {date.today().isoformat()}",
            "---",
            "",
            value.strip(),
            "",
        ]
    )
    topic.write_text(content, encoding="utf-8")

    index_title = _plain_scalar(title)
    index_description = _plain_scalar(value, limit=120)
    entry = f"- [{index_title}]({topic.name}) - {index_description}"
    index = paths.memory_index.read_text(encoding="utf-8", errors="ignore")
    lines = index.splitlines()
    link_target = f"]({topic.name})"
    for line_number, line in enumerate(lines):
        if link_target in line:
            lines[line_number] = entry
            break
    else:
        lines.append(entry)
    paths.memory_index.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json.dumps({"path": topic.as_posix(), "type": memory_type}, ensure_ascii=False)


def _frontmatter_scalar(text: str, limit: int | None = None) -> str:
    return json.dumps(_plain_scalar(text, limit=limit), ensure_ascii=False)


def _plain_scalar(text: str, limit: int | None = None) -> str:
    scalar = " ".join(text.strip().split())
    if limit is not None:
        scalar = scalar[:limit]
    return scalar.replace("---", "- - -")


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "memory"
