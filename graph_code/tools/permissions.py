"""Permission evaluation for tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"


READ_ONLY_TOOLS = {
    "read_file",
    "search_files",
    "task_get",
    "task_list",
    "background_check",
    "schedule_list",
    "load_skill",
    "compact",
}

WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "bash",
    "todo",
    "save_memory",
    "task_create",
    "task_update",
    "task_complete",
    "background_run",
    "schedule_create",
    "schedule_delete",
    "team_spawn",
    "send_message",
    "request_shutdown",
    "submit_plan_approval",
    "claim_task",
    "worktree_create",
    "worktree_enter",
    "worktree_run",
    "worktree_closeout",
}


@dataclass(frozen=True)
class PermissionDecision:
    action: str
    reason: str
    risk: str = "normal"

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    @property
    def denied(self) -> bool:
        return self.action == "deny"

    @property
    def ask(self) -> bool:
        return self.action == "ask"


FATAL_BASH_PATTERNS = [
    re.compile(r"\brm\s+-[^\n;]*[rf][^\n;]*\s+/(?:\s|$)"),
    re.compile(r"\brm\s+-[^\n;]*[rf][^\n;]*\s+/\*"),
    re.compile(r"\bdd\s+if="),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
]

SUSPICIOUS_BASH_PATTERNS = {
    "sudo": re.compile(r"(^|\s)sudo(\s|$)"),
    "rm_rf": re.compile(r"\brm\s+-[^\n;]*[rf]"),
    "command_substitution": re.compile(r"(`[^`]+`|\$\([^)]+\))"),
    "redirection": re.compile(r"(^|[^<])>{1,2}[^>]|<"),
    "shell_metacharacters": re.compile(r"(\|\||&&|;|\|)"),
}


def evaluate_permission(
    tool_call: dict[str, Any],
    mode: PermissionMode | str = PermissionMode.DEFAULT,
) -> PermissionDecision:
    """Evaluate one tool call using deny -> mode -> allow -> ask ordering."""
    mode = PermissionMode(mode)
    name = tool_call.get("name", "")
    args = tool_call.get("args", {}) or {}

    if name == "bash":
        command = str(args.get("command", ""))
        for pattern in FATAL_BASH_PATTERNS:
            if pattern.search(command):
                return PermissionDecision("deny", "Dangerous bash command blocked", "dangerous_command")

    if name.startswith("mcp__"):
        if mode == PermissionMode.AUTO:
            return PermissionDecision("allow", "Auto mode allows MCP tool", "mcp")
        return PermissionDecision("ask", "MCP tool requires approval", "mcp")

    if mode == PermissionMode.PLAN and name not in READ_ONLY_TOOLS:
        return PermissionDecision("ask", "Plan mode requires approval for side effects", "plan_mode")

    if mode == PermissionMode.AUTO:
        return PermissionDecision("allow", "Auto mode allows non-denied tool", "auto_mode")

    if name == "bash":
        command = str(args.get("command", ""))
        for risk, pattern in SUSPICIOUS_BASH_PATTERNS.items():
            if pattern.search(command):
                return PermissionDecision("ask", f"Bash command contains {risk}", "dangerous_command")

    if name in READ_ONLY_TOOLS:
        return PermissionDecision("allow", "Read-only tool", "read_only")

    if name in WRITE_TOOLS:
        return PermissionDecision("ask", "Side-effecting tool requires approval", "side_effect")

    return PermissionDecision("ask", "Unknown tool requires approval", "unknown_tool")


def build_permission_request(
    tool_call: dict[str, Any],
    decision: PermissionDecision,
) -> dict[str, Any]:
    return {
        "tool_call": tool_call,
        "tool_call_id": tool_call.get("id") or "unknown",
        "tool_name": tool_call.get("name", "unknown"),
        "args": tool_call.get("args", {}) or {},
        "reason": decision.reason,
        "risk": decision.risk,
    }
