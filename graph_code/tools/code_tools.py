"""Code analysis tools for Graph Code."""

import os
import re
from pathlib import Path
from typing import List, Optional

from ..config import get_config


def _get_safe_path(file_path: str) -> Path:
    """Get absolute path and ensure it's within working directory."""
    config = get_config()
    working_dir = config.working_path

    if os.path.isabs(file_path):
        target = Path(file_path).resolve()
    else:
        target = (working_dir / file_path).resolve()

    try:
        target.relative_to(working_dir)
    except ValueError:
        raise ValueError(f"Access denied: {file_path} is outside working directory")

    return target


def grep_search(pattern: str, path: str = ".", glob: Optional[str] = None) -> str:
    """Search for pattern in files using regex.

    Args:
        pattern: Regular expression pattern to search
        path: Directory or file to search in
        glob: Optional file glob pattern to filter (e.g., "*.py")

    Returns:
        Search results with file names and line numbers
    """
    try:
        target = _get_safe_path(path)
    except ValueError as e:
        return str(e)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    match_count = 0
    file_count = 0

    try:
        if target.is_file():
            files_to_search = [target]
        else:
            files_to_search = []
            for root, dirs, files in os.walk(target):
                for filename in files:
                    if glob and not fnmatch.fnmatch(filename, glob):
                        continue
                    full_path = os.path.join(root, filename)
                    files_to_search.append(Path(full_path))

        for file_path in files_to_search:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')

                file_matches = []
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        file_matches.append((line_num, line.strip()))

                if file_matches:
                    file_count += 1
                    rel_path = os.path.relpath(file_path, target) if target.is_dir() else str(file_path)
                    results.append(f"\n{rel_path}:")
                    for line_num, line_content in file_matches:
                        results.append(f"  {line_num:4d}: {line_content[:100]}")
                        match_count += 1

            except Exception:
                continue

    except Exception as e:
        return f"Error during search: {e}"

    if not results:
        return f"No matches found for pattern: {pattern}"

    summary = f"Found {match_count} matches in {file_count} files"
    return summary + "\n" + "=" * 60 + "".join(results)


def read_code_chunk(
    file_path: str,
    start_line: int,
    end_line: Optional[int] = None,
    context_lines: int = 3
) -> str:
    """Read a specific chunk of code with context.

    Args:
        file_path: Path to the file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        context_lines: Number of context lines before and after

    Returns:
        Code chunk with line numbers
    """
    try:
        target = _get_safe_path(file_path)
    except ValueError as e:
        return str(e)

    if not target.exists():
        return f"Error: File not found: {file_path}"

    if end_line is None:
        end_line = start_line

    try:
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = len(lines)

    # Calculate range with context
    display_start = max(0, start_line - context_lines - 1)
    display_end = min(total_lines, end_line + context_lines)

    # Calculate actual code range
    code_start = start_line - 1
    code_end = min(end_line, total_lines)

    result_lines = []
    result_lines.append(f"File: {file_path}")
    result_lines.append("=" * 60)

    for i in range(display_start, display_end):
        line_num = i + 1
        line_content = lines[i].rstrip()

        # Mark lines within the requested range
        if code_start <= i < code_end:
            prefix = f">>> {line_num:4d} | "
        else:
            prefix = f"    {line_num:4d} | "

        result_lines.append(prefix + line_content)

    return "\n".join(result_lines)


# Import fnmatch for glob pattern matching
import fnmatch
