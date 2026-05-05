"""Tool execution runtime and built-in tool implementations."""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..entities.background import background_check, background_run
from ..entities.schedules import schedule_create, schedule_delete, schedule_list
from ..entities.tasks import (
    claim_task,
    task_complete,
    task_create,
    task_get,
    task_list,
    task_update,
)
from ..entities.teams import (
    request_shutdown,
    send_message,
    submit_plan_approval,
    team_spawn,
)
from ..entities.worktrees import (
    worktree_closeout,
    worktree_create,
    worktree_enter,
    worktree_run,
)
from ..mcp.client import MCPClientRegistry
from .permissions import PermissionMode, evaluate_permission
from .schema import ToolResultEnvelope


READ_PARALLEL_TOOLS = {
    "read_file",
    "search_files",
    "task_get",
    "task_list",
    "background_check",
    "schedule_list",
}

SERIAL_TOOLS = {
    "write_file",
    "edit_file",
    "bash",
    "worktree_create",
    "worktree_enter",
    "worktree_run",
    "worktree_closeout",
}


class ToolExecutionRuntime:
    """Executes tool calls with uniform envelopes and output persistence."""

    def __init__(
        self,
        working_dir: str | Path,
        output_limit: int = 12000,
        mcp_registry: MCPClientRegistry | None = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.output_limit = output_limit
        self.agent_dir = self.working_dir / ".agent"
        self.output_dir = self.agent_dir / "tool-outputs"
        self.mcp_registry = mcp_registry or MCPClientRegistry(self.working_dir)

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        permission_mode: PermissionMode | str = PermissionMode.DEFAULT,
        skip_permissions: bool = False,
    ) -> list[ToolResultEnvelope]:
        """Execute calls and return results in the original order."""
        ordered: list[ToolResultEnvelope | None] = [None] * len(tool_calls)
        read_jobs: list[tuple[int, dict[str, Any]]] = []

        for index, call in enumerate(tool_calls):
            name = call.get("name", "")
            if not skip_permissions:
                decision = evaluate_permission(call, permission_mode)
                if decision.denied:
                    ordered[index] = self._denied(call, decision.reason)
                    continue
                if decision.ask:
                    ordered[index] = self._denied(call, f"Permission required: {decision.reason}")
                    continue

            if name in READ_PARALLEL_TOOLS:
                read_jobs.append((index, call))
            else:
                ordered[index] = self._execute_one(call)

        if read_jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(read_jobs))) as pool:
                futures = {
                    pool.submit(self._execute_one, call): index
                    for index, call in read_jobs
                }
                for future, index in futures.items():
                    ordered[index] = future.result()

        return [result for result in ordered if result is not None]

    def _execute_one(self, tool_call: dict[str, Any]) -> ToolResultEnvelope:
        tool_call_id = (tool_call.get("id") or "unknown").strip() or "unknown"
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}

        try:
            if name.startswith("mcp__"):
                result = self.mcp_registry.call_tool(name, args)
                result.tool_call_id = tool_call_id
                return self._persist_if_large(result, name)

            handlers: dict[str, Callable[..., str | dict[str, Any] | ToolResultEnvelope]] = {
                "read_file": self.read_file,
                "write_file": self.write_file,
                "edit_file": self.edit_file,
                "bash": self.bash,
                "search_files": self.search_files,
                "todo": self.todo,
                "load_skill": self.load_skill,
                "compact": self.compact,
                "save_memory": self.save_memory,
                "task_create": self._task_create,
                "task_update": self._task_update,
                "task_get": self._task_get,
                "task_list": self._task_list,
                "task_complete": self._task_complete,
                "background_run": self._background_run,
                "background_check": self._background_check,
                "schedule_create": self._schedule_create,
                "schedule_list": self._schedule_list,
                "schedule_delete": self._schedule_delete,
                "team_spawn": self._team_spawn,
                "send_message": self._send_message,
                "request_shutdown": self._request_shutdown,
                "submit_plan_approval": self._submit_plan_approval,
                "claim_task": self._claim_task,
                "worktree_create": self._worktree_create,
                "worktree_enter": self._worktree_enter,
                "worktree_run": self._worktree_run,
                "worktree_closeout": self._worktree_closeout,
                # Legacy aliases.
                "_read_file": self.read_file,
                "_write_file": self.write_file,
                "_glob_search": self.search_files,
                "_grep_search": self.search_files,
                "_bash_command": self.bash,
                "bash_command": self.bash,
                "grep_search": self.search_files,
                "glob_search": self.search_files,
            }
            if name not in handlers:
                return ToolResultEnvelope.error(
                    f"Unknown tool: {name}",
                    tool_call_id=tool_call_id,
                    metadata={"tool_name": name},
                )

            value = handlers[name](**args)
            if isinstance(value, ToolResultEnvelope):
                result = value
                result.tool_call_id = tool_call_id
            else:
                result = ToolResultEnvelope.success(
                    _stringify(value),
                    tool_call_id=tool_call_id,
                    metadata={"tool_name": name},
                )
            return self._persist_if_large(result, name)
        except Exception as exc:
            return ToolResultEnvelope.error(
                f"{type(exc).__name__}: {exc}",
                tool_call_id=tool_call_id,
                metadata={"tool_name": name},
            )

    def _denied(self, tool_call: dict[str, Any], reason: str) -> ToolResultEnvelope:
        return ToolResultEnvelope.error(
            f"Permission denied or blocked: {reason}",
            tool_call_id=(tool_call.get("id") or "unknown").strip() or "unknown",
            metadata={"tool_name": tool_call.get("name", "unknown"), "permission": "denied"},
        )

    def _persist_if_large(self, result: ToolResultEnvelope, tool_name: str) -> ToolResultEnvelope:
        result.metadata.setdefault("tool_name", tool_name)
        result.metadata.setdefault("persisted_output", None)
        if len(result.content) <= self.output_limit:
            return result

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{uuid.uuid4().hex}.txt"
        output_path = self.output_dir / output_name
        full_content = result.content
        output_path.write_text(full_content, encoding="utf-8")
        rel = output_path.relative_to(self.working_dir).as_posix()
        result.content = (
            full_content[: self.output_limit]
            + f"\n\n[persisted-output: {rel}]"
        )
        result.metadata["persisted_output"] = rel
        return result

    def _safe_path(self, file_path: str) -> Path:
        path = Path(file_path)
        target = path.resolve() if path.is_absolute() else (self.working_dir / path).resolve()
        try:
            target.relative_to(self.working_dir)
        except ValueError as exc:
            raise ValueError(f"Access denied: {file_path} is outside working directory") from exc
        return target

    def read_file(self, file_path: str, offset: int = 0, limit: int | None = None) -> str:
        target = self._safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        end = len(lines) if limit is None else min(len(lines), offset + limit)
        if offset >= len(lines):
            return f"File has {len(lines)} lines, offset {offset} is out of range"
        selected = lines[offset:end]
        rendered = [f"File: {file_path} (lines {offset + 1}-{end} of {len(lines)})"]
        rendered.extend(f"{line_no:4d} | {line}" for line_no, line in enumerate(selected, offset + 1))
        return "\n".join(rendered)

    def write_file(self, file_path: str, content: str, append: bool = False) -> str:
        target = self._safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return f"{'Appended to' if append else 'Wrote'} file: {file_path}"

    def edit_file(self, file_path: str, old: str, new: str, replace_all: bool = False) -> str:
        target = self._safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        original = target.read_text(encoding="utf-8", errors="ignore")
        count = original.count(old)
        if count == 0:
            return f"Error: Text not found in {file_path}"
        if count > 1 and not replace_all:
            return f"Error: Text occurs {count} times; set replace_all=true"
        edited = original.replace(old, new, -1 if replace_all else 1)
        target.write_text(edited, encoding="utf-8")
        diff = difflib.unified_diff(
            original.splitlines(),
            edited.splitlines(),
            fromfile=f"{file_path}:before",
            tofile=f"{file_path}:after",
            lineterm="",
        )
        return "\n".join(diff) or f"Edited file: {file_path}"

    def bash(self, command: str, timeout: int = 60) -> str:
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parts = [f"Command: {command}", f"Exit code: {result.returncode}"]
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append("STDERR:")
            parts.append(result.stderr)
        return "\n".join(parts)

    def search_files(
        self,
        pattern: str | None = None,
        path: str = ".",
        glob: str | None = None,
        query: str | None = None,
    ) -> str:
        needle = pattern or query or ""
        root = self._safe_path(path)
        if not root.exists():
            return f"Error: Path not found: {path}"
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        if glob:
            files = [p for p in files if p.match(glob) or p.name == glob or p.name.endswith(glob.lstrip("*"))]
        regex = re.compile(needle) if needle else None
        matches: list[str] = []
        for file_path in files:
            try:
                for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if regex is None or regex.search(line):
                        rel = file_path.relative_to(self.working_dir)
                        matches.append(f"{rel}:{line_no}: {line[:200]}")
            except OSError:
                continue
        return "\n".join(matches) if matches else f"No matches found for pattern: {needle}"

    def todo(self, items: list[dict[str, Any]] | None = None, action: str = "list") -> str:
        todo_path = self.agent_dir / "todo.json"
        todo_path.parent.mkdir(parents=True, exist_ok=True)
        current = json.loads(todo_path.read_text()) if todo_path.exists() else []
        if action == "set":
            validation_error = _validate_todo_items(items or [])
            if validation_error:
                return f"Error: {validation_error}"
            current = items or []
            todo_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return json.dumps(current, ensure_ascii=False)

    def load_skill(self, name: str, path: str | None = None) -> str:
        target = Path(path) if path else self.working_dir / ".agent" / "skills" / name / "SKILL.md"
        if not target.is_absolute():
            target = (self.working_dir / target).resolve()
        if not target.exists():
            return f"Error: Skill not found: {name}"
        return target.read_text(encoding="utf-8", errors="ignore")

    def compact(self, mode: str = "manual", summary: str | None = None) -> str:
        return json.dumps({"mode": mode, "summary": summary or ""})

    def save_memory(self, namespace: str, key: str, value: str) -> str:
        memory_dir = self.agent_dir / "memory" / namespace
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / f"{key}.json").write_text(json.dumps({"value": value}, indent=2), encoding="utf-8")
        return f"Saved memory: {namespace}/{key}"

    def _task_create(self, subject: str, description: str = "", **kwargs: Any) -> str:
        return task_create(self.working_dir, subject=subject, description=description, **kwargs).model_dump_json()

    def _task_update(self, task_id: str, **updates: Any) -> str:
        return task_update(self.working_dir, task_id, **updates).model_dump_json()

    def _task_get(self, task_id: str) -> str:
        return task_get(self.working_dir, task_id).model_dump_json()

    def _task_list(self) -> str:
        return json.dumps([task.model_dump() for task in task_list(self.working_dir)])

    def _task_complete(self, task_id: str) -> str:
        return task_complete(self.working_dir, task_id).model_dump_json()

    def _background_run(self, command: str, timeout: int = 3600) -> str:
        return background_run(self.working_dir, command=command, timeout=timeout).model_dump_json()

    def _background_check(self, runtime_task_id: str) -> str:
        return json.dumps(background_check(self.working_dir, runtime_task_id))

    def _schedule_create(self, cron: str, prompt: str, recurring: bool = True, durable: bool = True) -> str:
        return schedule_create(self.working_dir, cron, prompt, recurring, durable).model_dump_json()

    def _schedule_list(self) -> str:
        return json.dumps(schedule_list(self.working_dir))

    def _schedule_delete(self, schedule_id: str) -> str:
        return schedule_delete(self.working_dir, schedule_id).model_dump_json()

    def _team_spawn(self, name: str, role: str = "worker", prompt: str = "") -> str:
        return team_spawn(self.working_dir, name=name, role=role, prompt=prompt).model_dump_json()

    def _send_message(self, teammate_id: str, content: str, request_id: str | None = None) -> str:
        return send_message(self.working_dir, teammate_id, content, request_id).model_dump_json()

    def _request_shutdown(self, teammate_id: str, reason: str = "") -> str:
        return request_shutdown(self.working_dir, teammate_id, reason).model_dump_json()

    def _submit_plan_approval(self, request_id: str, approved: bool, notes: str = "") -> str:
        return submit_plan_approval(self.working_dir, request_id, approved, notes).model_dump_json()

    def _claim_task(self, task_id: str, owner: str) -> str:
        return claim_task(self.working_dir, task_id, owner).model_dump_json()

    def _worktree_create(self, task_id: str, base_path: str | None = None) -> str:
        return worktree_create(self.working_dir, task_id, Path(base_path) if base_path else self.working_dir).model_dump_json()

    def _worktree_enter(self, worktree_id: str) -> str:
        return worktree_enter(self.working_dir, worktree_id).model_dump_json()

    def _worktree_run(self, worktree_id: str, command: str, timeout: int = 60) -> str:
        return worktree_run(self.working_dir, worktree_id, command, timeout).model_dump_json()

    def _worktree_closeout(self, worktree_id: str, mode: str = "keep") -> str:
        return worktree_closeout(self.working_dir, worktree_id, mode).model_dump_json()


def _stringify(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _validate_todo_items(items: list[dict[str, Any]]) -> str | None:
    allowed = {"pending", "in_progress", "completed"}
    in_progress = 0
    for item in items:
        status = item.get("status", "pending")
        if status not in allowed:
            return f"invalid todo status: {status}"
        if status == "in_progress":
            in_progress += 1
    if in_progress > 1:
        return "todo list can have at most one in_progress item"
    return None
