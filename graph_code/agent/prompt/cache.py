"""Small prompt section cache helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def cached_section(state: dict[str, Any], name: str, compute: Callable[[], str | None]) -> str | None:
    prompt_state = state.setdefault("prompt_state", {})
    cache = prompt_state.setdefault("cache", {})
    if not prompt_state.get("invalidated") and name in cache:
        return cache[name]
    value = compute()
    cache[name] = value
    return value


def invalidate_prompt_cache(state: dict[str, Any]) -> dict[str, Any]:
    prompt_state = dict(state.get("prompt_state") or {})
    prompt_state["cache"] = {}
    prompt_state["invalidated"] = True
    state["prompt_state"] = prompt_state
    return prompt_state
