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
    root = Path(memory_dir)
    if not root.exists():
        return []
    headers: list[MemoryHeader] = []
    for path in root.rglob("*.md"):
        if path.name == "MEMORY.md" or not path.is_file():
            continue
        try:
            parsed = parse_frontmatter(path.read_text(encoding="utf-8", errors="ignore"))
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        headers.append(
            MemoryHeader(
                filename=rel,
                path=path,
                description=parsed.metadata.get("description"),
                memory_type=normalize_memory_type(parsed.metadata.get("type")),
                mtime=stat.st_mtime,
            )
        )
    return sorted(headers, key=lambda item: item.mtime, reverse=True)[:MAX_MEMORY_FILES]
