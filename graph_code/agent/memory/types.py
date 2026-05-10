"""Memory type taxonomy and frontmatter parsing."""

from __future__ import annotations

from dataclasses import dataclass

MEMORY_TYPES = {"user", "feedback", "project", "reference"}


@dataclass(frozen=True)
class ParsedFrontmatter:
    metadata: dict[str, str]
    body: str


def parse_frontmatter(content: str) -> ParsedFrontmatter:
    if not content.startswith("---\n"):
        return ParsedFrontmatter(metadata={}, body=content)
    end = content.find("\n---", 4)
    if end == -1:
        return ParsedFrontmatter(metadata={}, body=content)
    raw = content[4:end].strip()
    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    body = content[end + len("\n---") :].lstrip("\n")
    return ParsedFrontmatter(metadata=metadata, body=body)


def normalize_memory_type(raw: str | None) -> str | None:
    if raw in MEMORY_TYPES:
        return raw
    return None
