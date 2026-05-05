"""Persistent task records."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    id: str
    subject: str
    description: str = ""
    status: str = "pending"
    blockedBy: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    owner: str | None = None
    worktree: str | None = None
    created_at: str
    updated_at: str
    event_log: list[dict[str, Any]] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tasks_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(root: str | Path, task_id: str) -> Path:
    return _tasks_dir(root) / f"{task_id}.json"


def _save(root: str | Path, record: TaskRecord) -> TaskRecord:
    record.updated_at = _now()
    _path(root, record.id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def task_create(
    root: str | Path,
    subject: str,
    description: str = "",
    blocked_by: list[str] | None = None,
    blocks: list[str] | None = None,
    owner: str | None = None,
    worktree: str | None = None,
) -> TaskRecord:
    task_id = f"task-{uuid.uuid4().hex[:10]}"
    now = _now()
    record = TaskRecord(
        id=task_id,
        subject=subject,
        description=description,
        status="blocked" if blocked_by else "pending",
        blockedBy=blocked_by or [],
        blocks=blocks or [],
        owner=owner,
        worktree=worktree,
        created_at=now,
        updated_at=now,
        event_log=[{"event": "created", "at": now}],
    )
    _save(root, record)
    for blocker_id in record.blockedBy:
        try:
            blocker = task_get(root, blocker_id)
        except FileNotFoundError:
            continue
        if record.id not in blocker.blocks:
            blocker.blocks.append(record.id)
            blocker.event_log.append({"event": "blocks_added", "task_id": record.id, "at": _now()})
            _save(root, blocker)
    return record


def task_get(root: str | Path, task_id: str) -> TaskRecord:
    path = _path(root, task_id)
    if not path.exists():
        raise FileNotFoundError(task_id)
    return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))


def task_list(root: str | Path) -> list[TaskRecord]:
    return [
        TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(_tasks_dir(root).glob("*.json"))
    ]


def task_update(root: str | Path, task_id: str, **updates: Any) -> TaskRecord:
    record = task_get(root, task_id)
    for key, value in updates.items():
        if value is not None and hasattr(record, key):
            setattr(record, key, value)
    record.event_log.append({"event": "updated", "updates": updates, "at": _now()})
    return _save(root, record)


def task_complete(root: str | Path, task_id: str) -> TaskRecord:
    record = task_get(root, task_id)
    record.status = "completed"
    record.event_log.append({"event": "completed", "at": _now()})
    _save(root, record)
    for dependent_id in list(record.blocks):
        try:
            dependent = task_get(root, dependent_id)
        except FileNotFoundError:
            continue
        dependent.blockedBy = [bid for bid in dependent.blockedBy if bid != task_id]
        if not dependent.blockedBy and dependent.status == "blocked":
            dependent.status = "pending"
            dependent.event_log.append({"event": "unblocked", "by": task_id, "at": _now()})
        _save(root, dependent)
    return record


def claim_task(root: str | Path, task_id: str, owner: str) -> TaskRecord:
    lock = _tasks_dir(root) / f"{task_id}.lock"
    try:
        fd = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Task already claimed: {task_id}") from exc
    with fd:
        fd.write(json.dumps({"owner": owner, "at": _now()}))
    record = task_get(root, task_id)
    record.owner = owner
    record.status = "in_progress"
    record.event_log.append({"event": "claimed", "owner": owner, "at": _now()})
    return _save(root, record)
