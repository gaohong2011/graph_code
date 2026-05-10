"""Session memory prompt and template."""

DEFAULT_SESSION_MEMORY_TEMPLATE = """# Session Title
_A short and distinctive 5-10 word title._

# Current State
_What is actively being worked on right now?_

# Task Specification
_What did the user ask to build?_

# Files and Functions
_Important files, functions, and why they matter._

# Workflow
_Commands usually run and how to interpret them._

# Errors & Corrections
_Errors encountered, fixes, and user corrections._

# Codebase and System Documentation
_Important system components and how they fit together._

# Learnings
_What worked, what did not, what to avoid._

# Key Results
_Exact outputs or decisions the user asked for._

# Worklog
_Terse step-by-step work log._
"""

MAX_SESSION_WORKLOG_CHARS = 2000

SESSION_MEMORY_UPDATE_SYSTEM = (
    "Return a complete session memory markdown. Do not answer the user. "
    "Preserve exact identifiers, file paths, commands, decisions, errors, and user preferences. "
    "Use the provided section headings where applicable."
)


def build_mock_session_memory(messages_text: str) -> str:
    return DEFAULT_SESSION_MEMORY_TEMPLATE + "\n\n# Worklog\n" + messages_text[:2000] + "\n"


def build_session_memory_update_prompt(messages_text: str) -> str:
    return "\n\n".join(
        [
            "Update the session memory using this template:",
            DEFAULT_SESSION_MEMORY_TEMPLATE,
            "Recent transcript to preserve and summarize:",
            messages_text,
        ]
    )


def ensure_session_memory_worklog(content: str, messages_text: str) -> str:
    content = content.strip() or DEFAULT_SESSION_MEMORY_TEMPLATE.strip()
    excerpt = messages_text[-MAX_SESSION_WORKLOG_CHARS:].strip()
    if not excerpt or excerpt in content:
        return content + "\n"
    return content.rstrip() + "\n\n# Recent Transcript\n" + excerpt + "\n"
