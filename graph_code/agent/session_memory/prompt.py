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


def build_mock_session_memory(messages_text: str) -> str:
    return DEFAULT_SESSION_MEMORY_TEMPLATE + "\n\n# Current State\n" + messages_text[:2000] + "\n"
