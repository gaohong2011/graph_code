"""Best-effort turn-end session memory updater."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..compaction.policy import estimate_messages_tokens
from ..memory.paths import memory_paths_for_project
from ...llm.client import get_llm
from .prompt import build_mock_session_memory
from .state import should_update_session_memory


def _is_context_too_long_error(error: str) -> bool:
    normalized = error.lower()
    markers = (
        "context length",
        "context_length",
        "context too long",
        "prompt too long",
        "maximum context",
        "token limit",
        "too many tokens",
        "413",
    )
    return any(marker in normalized for marker in markers)


def maybe_update_session_memory(state: dict[str, Any], config: Any) -> dict[str, Any]:
    if not should_update_session_memory(state, config):
        return {}
    paths = memory_paths_for_project(config)
    paths.session_memory_dir.mkdir(parents=True, exist_ok=True)
    session_state = dict(state.get("session_memory_state") or {})
    text = _messages_text(state)
    try:
        if getattr(config, "llm_model", "mock") == "mock" or not getattr(config, "llm_api_key", None):
            content = build_mock_session_memory(text)
        else:
            response = get_llm(config=config).invoke(
                [
                    SystemMessage(content="Update the session memory markdown. Return markdown only. Do not call tools."),
                    HumanMessage(content=text),
                ]
            )
            content = str(getattr(response, "content", "")).strip() or build_mock_session_memory(text)
        paths.session_memory_file.write_text(content, encoding="utf-8")
        session_state.update(
            {
                "initialized": True,
                "path": paths.session_memory_file.as_posix(),
                "tokens_at_last_update": estimate_messages_tokens(list(state.get("messages", []))),
                "last_summarized_index": len(state.get("messages", [])),
                "last_error": None,
            }
        )
    except Exception as exc:
        if _is_context_too_long_error(str(exc)):
            session_state["last_error"] = "context_too_long"
        else:
            session_state["last_error"] = f"{type(exc).__name__}: {exc}"
    return {"session_memory_state": session_state}


def _messages_text(state: dict[str, Any]) -> str:
    lines = []
    for message in state.get("messages", [])[-40:]:
        lines.append(f"{getattr(message, 'type', type(message).__name__)}: {getattr(message, 'content', '')}")
    return "\n".join(lines)
