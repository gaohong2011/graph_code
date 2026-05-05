"""Prompts and formatting for model-assisted context compaction."""

from __future__ import annotations

import re
from typing import Any


COMPACT_SUMMARY_SECTIONS = [
    "Primary Request and Intent",
    "Key Technical Concepts",
    "Files and Code Sections",
    "Errors and fixes",
    "Problem Solving",
    "All user messages",
    "Pending Tasks",
    "Current Work",
    "Optional Next Step",
]


def build_model_compact_prompt(extractive_summary: str, *, short: bool = False) -> str:
    """Build a no-tools summary prompt inspired by Claude Code's compact prompt."""

    if short:
        return (
            "Create a concise continuation summary. Return plain text only. "
            "Preserve current goal, files, decisions, errors, current work, and next step.\n\n"
            f"State:\n{extractive_summary[:3000]}"
        )

    sections = "\n".join(f"{index}. {section}" for index, section in enumerate(COMPACT_SUMMARY_SECTIONS, 1))
    return (
        "Your task is to create a detailed summary of the coding-agent conversation so far. "
        "This summary will be used to continue development work after context compaction.\n\n"
        "Your entire response must be plain text. You may include an <analysis> block followed "
        "by a <summary> block; the analysis block will be stripped before the summary is added "
        "to model context.\n\n"
        "Do not request tools. Tool calls are unavailable during compaction.\n\n"
        "Your summary must include these sections:\n"
        f"{sections}\n\n"
        "Be concrete. Preserve filenames, code sections, commands, errors, fixes, user feedback, "
        "pending tasks, current work, and the next step that directly follows from the latest request.\n\n"
        f"Extractive state:\n{extractive_summary}"
    )


def format_model_compact_summary(raw: Any) -> str:
    """Strip analysis scratchpad and unwrap summary tags."""

    text = raw if isinstance(raw, str) else str(raw)
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"</?summary>", "", text, flags=re.IGNORECASE).strip()
    return text
