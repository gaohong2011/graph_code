"""Minimal teammate and protocol message records."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class TeammateRecord(BaseModel):
    id: str
    name: str
    role: str
    status: str = "idle"
    thread_id: str
    inbox: list[dict] = Field(default_factory=list)


class RequestRecord(BaseModel):
    id: str
    teammate_id: str | None = None
    kind: str
    status: str
    content: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _teams_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "teams"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _requests_dir(root: str | Path) -> Path:
    path = Path(root) / ".agent" / "requests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_team(root: str | Path, teammate: TeammateRecord) -> TeammateRecord:
    (_teams_dir(root) / f"{teammate.id}.json").write_text(
        teammate.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return teammate


def _load_team(root: str | Path, teammate_id: str) -> TeammateRecord:
    path = _teams_dir(root) / f"{teammate_id}.json"
    if not path.exists():
        raise FileNotFoundError(teammate_id)
    return TeammateRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _save_request(root: str | Path, request: RequestRecord) -> RequestRecord:
    (_requests_dir(root) / f"{request.id}.json").write_text(
        request.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return request


def team_spawn(root: str | Path, name: str, role: str = "worker", prompt: str = "") -> TeammateRecord:
    teammate_id = f"tm-{uuid.uuid4().hex[:10]}"
    teammate = TeammateRecord(
        id=teammate_id,
        name=name,
        role=role,
        thread_id=f"teammate-{teammate_id}",
        inbox=[{"request_id": f"req-{uuid.uuid4().hex[:10]}", "content": prompt}] if prompt else [],
    )
    return _save_team(root, teammate)


def send_message(
    root: str | Path,
    teammate_id: str,
    content: str,
    request_id: str | None = None,
) -> RequestRecord:
    teammate = _load_team(root, teammate_id)
    request = RequestRecord(
        id=request_id or f"req-{uuid.uuid4().hex[:10]}",
        teammate_id=teammate_id,
        kind="message",
        status="queued",
        content=content,
        created_at=_now(),
    )
    teammate.inbox.append({"request_id": request.id, "content": content})
    _save_team(root, teammate)
    return _save_request(root, request)


def request_shutdown(root: str | Path, teammate_id: str, reason: str = "") -> RequestRecord:
    return send_message(root, teammate_id, f"shutdown_request: {reason}")


def submit_plan_approval(
    root: str | Path,
    request_id: str,
    approved: bool,
    notes: str = "",
) -> RequestRecord:
    request = RequestRecord(
        id=request_id,
        kind="plan_approval",
        status="approved" if approved else "denied",
        content=json.dumps({"approved": approved, "notes": notes}),
        created_at=_now(),
    )
    return _save_request(root, request)
