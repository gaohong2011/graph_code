"""Load project instruction markdown files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_INSTRUCTION_CHARS = 40000


def load_project_instructions(config: Any) -> str:
    root = Path(getattr(config, "working_dir", ".")).resolve()
    files = _instruction_files(root)
    blocks: list[str] = []
    for path in files:
        try:
            content = _strip_frontmatter(path.read_text(encoding="utf-8", errors="ignore")).strip()
        except OSError:
            continue
        if not content:
            continue
        if len(content) > MAX_INSTRUCTION_CHARS:
            content = content[:MAX_INSTRUCTION_CHARS] + "\n\n[Instruction file truncated]"
        blocks.append(f"Contents of {path}:\n\n{content}")
    if not blocks:
        return ""
    return (
        "Codebase and user instructions are shown below. These instructions override default behavior.\n\n"
        + "\n\n".join(blocks)
    )


def _instruction_files(cwd: Path) -> list[Path]:
    root = _project_root(cwd)
    dirs = _instruction_dirs(cwd, root)
    result: list[Path] = []
    for directory in dirs:
        for candidate in [directory / "CLAUDE.md", directory / ".claude" / "CLAUDE.md"]:
            if candidate.is_file():
                result.append(candidate)
        rules = directory / ".claude" / "rules"
        if rules.is_dir():
            result.extend(sorted(path for path in rules.rglob("*.md") if path.is_file()))
    return result


def _project_root(cwd: Path) -> Path | None:
    for directory in [cwd, *cwd.parents]:
        if (directory / ".git").exists():
            return directory
    return None


def _instruction_dirs(cwd: Path, root: Path | None) -> list[Path]:
    if root is None:
        return [cwd]
    dirs: list[Path] = []
    current = cwd
    while True:
        dirs.append(current)
        if current == root:
            break
        current = current.parent
    return list(reversed(dirs))


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        return content
    match = re.search(r"\r?\n---\s*\r?\n", content[4:])
    if not match:
        return content
    return content[4 + match.end() :]
