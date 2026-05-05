"""Durable schedule records."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class ScheduleRecord(BaseModel):
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool
    created_at: str
    last_fired_at: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schedules_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "schedules"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(root: str | Path, schedule_id: str) -> Path:
    return _schedules_dir(root) / f"{schedule_id}.json"


def _save(root: str | Path, record: ScheduleRecord) -> ScheduleRecord:
    _path(root, record.id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def schedule_create(
    root: str | Path,
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> ScheduleRecord:
    record = ScheduleRecord(
        id=f"sched-{uuid.uuid4().hex[:10]}",
        cron=cron,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
        created_at=_now(),
    )
    return _save(root, record)


def schedule_list(root: str | Path, now: str | None = None) -> list[dict]:
    current = now or _now()
    notifications: list[dict] = []
    for path in sorted(_schedules_dir(root).glob("*.json")):
        record = ScheduleRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.last_fired_at is None:
            notifications.append(
                {
                    "type": "schedule_due",
                    "schedule_id": record.id,
                    "prompt": record.prompt,
                    "durable": record.durable,
                }
            )
            record.last_fired_at = current
            _save(root, record)
    return notifications


def schedule_delete(root: str | Path, schedule_id: str) -> ScheduleRecord:
    path = _path(root, schedule_id)
    if not path.exists():
        raise FileNotFoundError(schedule_id)
    record = ScheduleRecord.model_validate_json(path.read_text(encoding="utf-8"))
    path.unlink()
    return record
