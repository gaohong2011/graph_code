"""Worktree registry and execution helpers."""

from __future__ import annotations

import shutil
import subprocess
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class WorktreeDirtyError(RuntimeError):
    """Raised when removing a dirty worktree would discard changes."""


class WorktreeRecord(BaseModel):
    id: str
    task_id: str
    base_path: str
    path: str
    status: str = "active"
    created_at: str
    event_log: list[dict[str, Any]] = Field(default_factory=list)


class WorktreeRunRecord(BaseModel):
    worktree_id: str
    command: str
    returncode: int
    stdout: str
    stderr: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "worktrees"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_path(root: str | Path, worktree_id: str) -> Path:
    return _registry_dir(root) / f"{worktree_id}.json"


def _save(root: str | Path, record: WorktreeRecord) -> WorktreeRecord:
    _record_path(root, record.id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _load(root: str | Path, worktree_id: str) -> WorktreeRecord:
    path = _record_path(root, worktree_id)
    if not path.exists():
        raise FileNotFoundError(worktree_id)
    return WorktreeRecord.model_validate_json(path.read_text(encoding="utf-8"))


def worktree_create(root: str | Path, task_id: str, base_path: str | Path) -> WorktreeRecord:
    worktree_id = f"wt-{uuid.uuid4().hex[:10]}"
    path = _registry_dir(root) / worktree_id
    base = Path(base_path).resolve()
    event_log = [{"event": "created", "at": _now()}]
    if _is_git_repo(base):
        branch = _branch_name(task_id, worktree_id)
        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch, path.as_posix(), "HEAD"],
            cwd=base,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
        event_log.append({"event": "git_worktree_add", "branch": branch, "at": _now()})
    else:
        path.mkdir(parents=True, exist_ok=False)
        event_log.append({"event": "registry_directory_created", "at": _now()})
    record = WorktreeRecord(
        id=worktree_id,
        task_id=task_id,
        base_path=base.as_posix(),
        path=path.as_posix(),
        created_at=_now(),
        event_log=event_log,
    )
    return _save(root, record)


def worktree_enter(root: str | Path, worktree_id: str) -> WorktreeRecord:
    record = _load(root, worktree_id)
    record.event_log.append({"event": "entered", "at": _now()})
    return _save(root, record)


def _is_dirty(path: Path) -> bool:
    git_dir = path / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    return any(item.name != ".git" for item in path.iterdir())


def worktree_run(root: str | Path, worktree_id: str, command: str, timeout: int = 60) -> WorktreeRunRecord:
    record = _load(root, worktree_id)
    result = subprocess.run(
        command,
        shell=True,
        cwd=record.path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return WorktreeRunRecord(
        worktree_id=worktree_id,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def worktree_closeout(root: str | Path, worktree_id: str, mode: str = "keep") -> WorktreeRecord:
    record = _load(root, worktree_id)
    path = Path(record.path)
    if mode == "remove":
        if path.exists() and _is_dirty(path):
            raise WorktreeDirtyError(f"Worktree has uncommitted changes: {record.path}")
        if _is_git_repo(path):
            result = subprocess.run(
                ["git", "worktree", "remove", path.as_posix()],
                cwd=record.base_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git worktree remove failed: {result.stderr.strip()}")
        else:
            shutil.rmtree(path, ignore_errors=True)
        record.status = "removed"
    else:
        record.status = "kept"
    record.event_log.append({"event": "closeout", "mode": mode, "at": _now()})
    return _save(root, record)


def _is_git_repo(path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _branch_name(task_id: str, worktree_id: str) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9._/-]+", "-", task_id).strip("-") or "task"
    return f"graph-code/{safe_task}-{worktree_id}"
