"""Optional model-assisted relevant memory recall."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...llm.client import get_llm
from .paths import memory_paths_for_project
from .scan import scan_memory_headers


def select_relevant_memories(query: str, config: Any, limit: int = 5) -> list[Path]:
    if not getattr(config, "memory_relevance_enabled", False):
        return []
    headers = scan_memory_headers(memory_paths_for_project(config).memory_dir)
    if not headers:
        return []
    manifest = "\n".join(
        f"- {item.filename}: {item.description or ''} [{item.memory_type or 'unknown'}]"
        for item in headers
    )
    valid = {item.filename: item.path for item in headers}
    if getattr(config, "llm_model", "mock") == "mock" or not getattr(config, "llm_api_key", None):
        selected = [item.path for item in headers if query.lower() in item.filename.lower()]
        return selected[:limit]
    try:
        response = get_llm(config=config).invoke(
            [
                SystemMessage(
                    content=(
                        "Select up to five memory filenames that are clearly relevant. "
                        'Return JSON only: {"selected_memories": ["file.md"]}.'
                    )
                ),
                HumanMessage(content=f"Query: {query}\n\nAvailable memories:\n{manifest}"),
            ]
        )
        payload = json.loads(str(getattr(response, "content", "{}")))
    except Exception:
        return []
    names = payload.get("selected_memories") if isinstance(payload, dict) else []
    if not isinstance(names, list):
        return []
    return [valid[name] for name in names[:limit] if isinstance(name, str) and name in valid]


def build_relevant_memory_context(paths: list[Path]) -> str:
    if not paths:
        return ""
    blocks = ["Relevant memories:"]
    for path in paths[:5]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        blocks.append(f"## {path.name}\n{content[:8000]}")
    return "\n\n".join(blocks)
