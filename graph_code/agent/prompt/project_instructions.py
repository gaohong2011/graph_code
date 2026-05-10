"""Load project instruction markdown files."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from ..memory.paths import memory_paths_for_project

MAX_INSTRUCTION_CHARS = 40000
MAX_INCLUDE_DEPTH = 5
MAX_INCLUDE_FILES = 20
INCLUDE_EXTENSIONS = {".md", ".markdown", ".txt"}


def load_project_instructions(config: Any, active_paths: list[str] | None = None) -> str:
    cwd = Path(getattr(config, "working_dir", ".")).resolve()
    project_root = _project_root(cwd) or cwd
    memory_root = _memory_root(config)
    files = _instruction_files(cwd)
    active = _active_paths(active_paths or [], project_root, cwd)
    blocks: list[str] = []
    include_count = 0
    for path in files:
        loaded, include_count = _load_instruction_file(
            path,
            project_root=project_root,
            memory_root=memory_root,
            active_paths=active,
            seen=set(),
            include_count=include_count,
            depth=0,
        )
        if not loaded:
            continue
        blocks.append(f"Contents of {path}:\n\n{loaded}")
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
    return _parse_frontmatter(content)[1]


def _load_instruction_file(
    path: Path,
    *,
    project_root: Path,
    memory_root: Path | None,
    active_paths: list[str],
    seen: set[Path],
    include_count: int,
    depth: int,
) -> tuple[str | None, int]:
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return None, include_count
    if resolved in seen:
        return None, include_count
    if not _allowed_include_path(resolved, project_root, memory_root):
        return None, include_count
    try:
        raw = resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, include_count
    metadata, body = _parse_frontmatter(raw)
    patterns = _frontmatter_paths(metadata)
    if patterns is not None and not _matches_active_path(patterns, active_paths):
        return None, include_count
    seen.add(resolved)
    included_blocks: list[str] = []
    if depth < MAX_INCLUDE_DEPTH:
        for include_path in _include_paths(body, resolved.parent):
            if include_count >= MAX_INCLUDE_FILES:
                break
            if include_path.suffix.lower() not in INCLUDE_EXTENSIONS:
                continue
            loaded, include_count = _load_instruction_file(
                include_path,
                project_root=project_root,
                memory_root=memory_root,
                active_paths=active_paths,
                seen=seen,
                include_count=include_count + 1,
                depth=depth + 1,
            )
            if loaded:
                included_blocks.append(loaded)
    content = "\n\n".join([*included_blocks, body.strip()]).strip()
    if len(content) > MAX_INSTRUCTION_CHARS:
        content = content[:MAX_INSTRUCTION_CHARS] + "\n\n[Instruction file truncated]"
    return content or None, include_count


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        return {}, content
    match = re.search(r"\r?\n---\s*\r?\n", content[4:])
    if not match:
        return {}, content
    raw = content[4 : 4 + match.start()].strip()
    body = content[4 + match.end() :]
    return _parse_frontmatter_metadata(raw), body


def _parse_frontmatter_metadata(raw: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    lines = raw.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if ":" not in line:
            index += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value:
            metadata[key] = value
            index += 1
            continue
        items: list[str] = []
        index += 1
        while index < len(lines) and lines[index].startswith((" ", "\t", "-")):
            item = lines[index].strip()
            if item.startswith("-"):
                item = item[1:].strip()
            if item:
                items.append(item.strip('"').strip("'"))
            index += 1
        metadata[key] = ",".join(items)
    return metadata


def _frontmatter_paths(metadata: dict[str, str]) -> list[str] | None:
    raw = metadata.get("paths")
    if not raw:
        return None
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    paths = [
        item.strip().strip('"').strip("'")
        for item in re.split(r"[,\n]+", value)
        if item.strip()
    ]
    normalized = [path[:-3] if path.endswith("/**") else path for path in paths]
    normalized = [path for path in normalized if path and path != "**"]
    return normalized or None


def _active_paths(paths: list[str], project_root: Path, working_dir: Path) -> list[str]:
    active: list[str] = []
    for raw in paths:
        try:
            path = Path(raw)
            resolved = path.resolve() if path.is_absolute() else (working_dir / path).resolve()
            if _is_relative_to(resolved, project_root):
                active.append(resolved.relative_to(project_root).as_posix())
            else:
                active.append(raw.replace("\\", "/").lstrip("./"))
        except (OSError, ValueError):
            active.append(raw.replace("\\", "/").lstrip("./"))
    return active


def _matches_active_path(patterns: list[str], active_paths: list[str]) -> bool:
    if not active_paths:
        return False
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").lstrip("./")
        if normalized in {"", "**"}:
            return True
        for active in active_paths:
            if _matches_glob(active, normalized):
                return True
    return False


def _matches_glob(path: str, pattern: str) -> bool:
    if path == pattern or path.startswith(pattern.rstrip("/") + "/"):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    if fnmatch.fnmatch(path, pattern):
        return True
    if "/**/" in pattern and fnmatch.fnmatch(path, pattern.replace("/**/", "/")):
        return True
    if "/" not in pattern and fnmatch.fnmatch(Path(path).name, pattern):
        return True
    return False


def _include_paths(content: str, base_dir: Path) -> list[Path]:
    results: list[Path] = []
    include_regex = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")
    for match in include_regex.finditer(content):
        raw = match.group(1).split("#", 1)[0].replace("\\ ", " ").strip()
        if not raw or raw.startswith("@") or re.match(r"^[#%^&*()]+", raw):
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        results.append(path)
    return results


def _memory_root(config: Any) -> Path | None:
    if getattr(config, "memory_disabled", False):
        return None
    try:
        return memory_paths_for_project(config).memory_dir.resolve()
    except (OSError, ValueError):
        return None


def _allowed_include_path(path: Path, project_root: Path, memory_root: Path | None) -> bool:
    if _is_relative_to(path, project_root):
        return True
    if memory_root is not None and _is_relative_to(path, memory_root):
        return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
