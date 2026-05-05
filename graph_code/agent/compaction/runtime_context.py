"""Runtime context rehydration for post-compaction model input."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


def write_transcript(
    messages: list[BaseMessage],
    working_dir: str | Path,
    boundary_id: str,
) -> str:
    """Persist the full pre-compact transcript and return a workspace-relative path."""

    root = Path(working_dir).resolve()
    transcript_dir = root / ".agent" / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    rel = Path(".agent") / "transcripts" / f"{boundary_id}.jsonl"
    target = root / rel
    with target.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(_message_record(message), ensure_ascii=False, default=str))
            handle.write("\n")
    return rel.as_posix()


def run_compact_hook(
    working_dir: str | Path,
    name: str,
    *,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """Run an optional local compact hook and return structured output."""

    root = Path(working_dir).resolve()
    script = root / ".agent" / "hooks" / f"{name}.py"
    if not script.exists():
        return []
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        content = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        return [
            {
                "hook": name,
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "content": content,
            }
        ]
    except Exception as exc:
        return [
            {
                "hook": name,
                "ok": False,
                "exit_code": None,
                "content": f"{type(exc).__name__}: {exc}",
            }
        ]


def build_rehydration_text(
    state: dict[str, Any],
    *,
    transcript_path: str | None,
    post_compact_hooks: list[dict[str, Any]] | None = None,
) -> str:
    """Render compact post-attachments that should survive summary replacement."""

    lines = ["Runtime context after compaction:"]
    if state.get("current_task_id"):
        lines.append(f"- Current task: {state['current_task_id']}")
    planning = state.get("planning_state") or {}
    if planning and planning.get("status") != "none":
        lines.append(f"- Planning state: {_json_preview(planning)}")
    loaded_skills = state.get("loaded_skills") or {}
    if loaded_skills:
        lines.append(f"- Loaded skills manifest: {_json_preview(loaded_skills)}")
    worktree = state.get("worktree_context") or {}
    if worktree and (worktree.get("current") or worktree.get("registry")):
        lines.append(f"- Worktree context: {_json_preview(worktree)}")
    mcp_state = state.get("mcp_connection_state") or {}
    if mcp_state:
        lines.append(f"- MCP connection state: {_json_preview(mcp_state)}")
    notifications = state.get("notifications") or []
    if notifications:
        lines.append(f"- Notifications: {_json_preview(notifications[:5])}")
    if transcript_path:
        lines.append(f"- Full transcript: {transcript_path}")
    for hook in post_compact_hooks or []:
        lines.append(f"- PostCompact hook {hook.get('hook')}: {hook.get('content', '')}")
    return "\n".join(lines)


def _message_record(message: BaseMessage) -> dict[str, Any]:
    return {
        "type": getattr(message, "type", type(message).__name__),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def _json_preview(value: Any, limit: int = 2000) -> str:
    rendered = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."
