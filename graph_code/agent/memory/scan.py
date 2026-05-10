"""Scan Graph Code memory topic files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import normalize_memory_type, parse_frontmatter

MAX_MEMORY_FILES = 200


@dataclass(frozen=True)
class MemoryHeader:
    filename: str
    path: Path
    description: str | None
    memory_type: str | None
    mtime: float


def scan_memory_headers(memory_dir: str | Path) -> list[MemoryHeader]:
    try:
        root = Path(memory_dir).resolve()
    except (OSError, ValueError):
        return []
    if not root.is_dir():
        return []
    headers: list[MemoryHeader] = []
    for path in root.rglob("*.md"):
        if path.name == "MEMORY.md" or path.is_symlink():
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            parsed = parse_frontmatter(resolved.read_text(encoding="utf-8", errors="ignore"))
            stat = resolved.stat()
        except OSError:
            continue
        rel = resolved.relative_to(root).as_posix()
        headers.append(
            MemoryHeader(
                filename=rel,
                path=resolved,
                description=parsed.metadata.get("description"),
                memory_type=normalize_memory_type(parsed.metadata.get("type")),
                mtime=stat.st_mtime,
            )
        )
    return sorted(headers, key=lambda item: item.mtime, reverse=True)[:MAX_MEMORY_FILES]
