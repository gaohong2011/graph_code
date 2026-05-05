"""Background runtime tasks."""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

_PROCESSES: dict[str, subprocess.Popen] = {}


class RuntimeTaskRecord(BaseModel):
    id: str
    command: str
    status: str
    output_path: str
    pid: int | None = None
    returncode: int | None = None
    created_at: str
    completed_at: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "runtime-tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_path(root: str | Path, task_id: str) -> Path:
    return _runtime_dir(root) / f"{task_id}.json"


def _save(root: str | Path, record: RuntimeTaskRecord) -> RuntimeTaskRecord:
    _record_path(root, record.id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def background_run(root: str | Path, command: str, timeout: int = 3600) -> RuntimeTaskRecord:
    """Start a command and return immediately with a task record."""
    root = Path(root)
    task_id = f"rt-{uuid.uuid4().hex[:10]}"
    output_path = _runtime_dir(root) / f"{task_id}.out"
    handle = output_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handle.close()
    _PROCESSES[task_id] = process
    return _save(
        root,
        RuntimeTaskRecord(
            id=task_id,
            command=command,
            status="running",
            output_path=output_path.as_posix(),
            pid=process.pid,
            created_at=_now(),
        ),
    )


def background_check(root: str | Path, runtime_task_id: str) -> dict | None:
    path = _record_path(root, runtime_task_id)
    if not path.exists():
        raise FileNotFoundError(runtime_task_id)
    record = RuntimeTaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.status == "completed":
        return {
            "type": "runtime_task_completed",
            "runtime_task_id": record.id,
            "output_path": record.output_path,
            "returncode": record.returncode,
        }
    process = _PROCESSES.get(record.id)
    if process is not None:
        returncode = process.poll()
        if returncode is None:
            try:
                returncode = process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                return None
        record.returncode = returncode
    else:
        proc = subprocess.run(
            f"ps -p {record.pid}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return None
        record.returncode = 0
    record.status = "completed"
    record.completed_at = _now()
    _save(root, record)
    return {
        "type": "runtime_task_completed",
        "runtime_task_id": record.id,
        "output_path": record.output_path,
        "returncode": record.returncode,
    }
