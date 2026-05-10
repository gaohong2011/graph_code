"""Path resolution for Graph Code global project memory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryPaths:
    memory_dir: Path
    memory_index: Path
    session_memory_dir: Path
    session_memory_file: Path


def memory_paths_for_project(config: Any) -> MemoryPaths:
    override = getattr(config, "memory_dir", None)
    if override:
        root = validate_memory_root(override)
        if root is None:
            root = _default_memory_dir(config)
    else:
        root = _default_memory_dir(config)
    return MemoryPaths(
        memory_dir=root,
        memory_index=root / "MEMORY.md",
        session_memory_dir=root.parent / "session-memory",
        session_memory_file=root.parent / "session-memory" / "session.md",
    )


def validate_memory_root(raw: str | None) -> Path | None:
    if not raw:
        return None
    if "\0" in raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        path = candidate.resolve()
    except (OSError, ValueError):
        return None
    text = str(path)
    if "\0" in text:
        return None
    if path == path.anchor or len(path.parts) < 3:
        return None
    if path == Path.home().resolve():
        return None
    if path.exists() and not path.is_dir():
        return None
    return path


def _default_memory_dir(config: Any) -> Path:
    project = _canonical_project_root(Path(getattr(config, "working_dir", ".")).resolve())
    slug = _project_slug(project)
    return Path(getattr(config, "graph_code_home")).expanduser().resolve() / "projects" / slug / "memory"


def _canonical_project_root(path: Path) -> Path:
    for directory in [path, *path.parents]:
        if (directory / ".git").exists():
            return directory.resolve()
    return path


def _project_slug(path: Path) -> str:
    normalized = path.as_posix()
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized.strip("/"))[:80].strip("-")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{base or 'project'}-{digest}"
