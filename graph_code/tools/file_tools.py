"""File operation tools for Graph Code."""

import fnmatch
import os
from pathlib import Path
from typing import List, Optional

from ..config import get_config


def _get_safe_path(file_path: str) -> Path:
    """Get absolute path and ensure it's within working directory."""
    config = get_config()
    working_dir = config.working_path

    # Resolve to absolute path
    if os.path.isabs(file_path):
        target = Path(file_path).resolve()
    else:
        target = (working_dir / file_path).resolve()

    # Security check: ensure path is within working directory
    try:
        target.relative_to(working_dir)
    except ValueError:
        raise ValueError(f"Access denied: {file_path} is outside working directory")

    return target


def read_file(file_path: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Read content from a file.

    Args:
        file_path: Path to the file (relative to working directory or absolute)
        offset: Line number to start reading from (0-indexed)
        limit: Maximum number of lines to read

    Returns:
        File content as string with line numbers
    """
    try:
        target = _get_safe_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Error: File not found: {file_path}"

    if not target.is_file():
        return f"Error: Not a file: {file_path}"

    try:
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    # Apply offset and limit
    total_lines = len(lines)
    start = offset
    end = total_lines if limit is None else min(offset + limit, total_lines)

    if start >= total_lines:
        return f"File has {total_lines} lines, offset {offset} is out of range"

    selected_lines = lines[start:end]

    # Format with line numbers
    result_lines = []
    for i, line in enumerate(selected_lines, start=start + 1):
        result_lines.append(f"{i:4d} | {line.rstrip()}")

    header = f"File: {file_path} (lines {start+1}-{end} of {total_lines})\n"
    header += "-" * 60 + "\n"

    return header + "\n".join(result_lines)


def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file (relative to working directory or absolute)
        content: Content to write
        append: If True, append to file; if False, overwrite

    Returns:
        Success or error message
    """
    try:
        target = _get_safe_path(file_path)
    except ValueError as e:
        return str(e)

    try:
        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append else 'w'
        with open(target, mode, encoding='utf-8') as f:
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return f"{action} file: {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(dir_path: str = ".", recursive: bool = False) -> str:
    """List directory contents.

    Args:
        dir_path: Path to directory (relative to working directory or absolute)
        recursive: If True, list recursively

    Returns:
        Directory listing as formatted string
    """
    try:
        target = _get_safe_path(dir_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Error: Directory not found: {dir_path}"

    if not target.is_dir():
        return f"Error: Not a directory: {dir_path}"

    try:
        lines = [f"Directory: {dir_path}", "=" * 60]

        if recursive:
            for root, dirs, files in os.walk(target):
                level = root.replace(str(target), '').count(os.sep)
                indent = '  ' * level
                rel_path = os.path.relpath(root, target)
                if rel_path == '.':
                    rel_path = ''
                lines.append(f"{indent}{rel_path or '.'}/")
                for file in sorted(files):
                    lines.append(f"{indent}  {file}")
        else:
            items = sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item in items:
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                size = ""
                if item.is_file():
                    try:
                        size_bytes = item.stat().st_size
                        if size_bytes < 1024:
                            size = f" {size_bytes}B"
                        elif size_bytes < 1024 * 1024:
                            size = f" {size_bytes // 1024}KB"
                        else:
                            size = f" {size_bytes // (1024 * 1024)}MB"
                    except:
                        pass
                lines.append(f"{prefix} {item.name}{size}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


def glob_search(pattern: str, dir_path: str = ".") -> str:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "*.txt")
        dir_path: Directory to search in

    Returns:
        List of matching files
    """
    try:
        target = _get_safe_path(dir_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Error: Directory not found: {dir_path}"

    try:
        matches = []
        for root, dirs, files in os.walk(target):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, target)
                # Check against both relative path and filename
                # Also handle **/ patterns by matching basename
                if fnmatch.fnmatch(rel_path, pattern) or \
                   fnmatch.fnmatch(filename, pattern) or \
                   fnmatch.fnmatch(os.path.basename(rel_path), pattern.replace("**/", "")):
                    matches.append(rel_path)

        if not matches:
            return f"No files found matching pattern: {pattern}"

        lines = [f"Files matching '{pattern}':", "=" * 60]
        for match in sorted(matches):
            lines.append(match)

        return "\n".join(lines)
    except Exception as e:
        return f"Error searching files: {e}"
