"""Human interaction tools for Graph Code."""

from typing import Optional


class InteractionStore:
    """Store for pending user interactions."""

    def __init__(self):
        self.pending_question: Optional[str] = None
        self.pending_confirmation: Optional[dict] = None
        self.last_answer: Optional[str] = None

    def clear(self):
        """Clear all pending interactions."""
        self.pending_question = None
        self.pending_confirmation = None


# Global store instance
_store = InteractionStore()


def get_interaction_store() -> InteractionStore:
    """Get the global interaction store."""
    return _store


def ask_user(question: str) -> str:
    """Ask the user a question and wait for their answer.

    This tool sets up a pending question that will be shown to the user.
    The agent will pause until the user provides an answer.

    Args:
        question: The question to ask the user

    Returns:
        A message indicating the question is pending
    """
    store = get_interaction_store()
    store.pending_question = question
    return f"PENDING_QUESTION: {question}"


def confirm_action(action: str, details: str = "") -> str:
    """Request user confirmation before executing a sensitive action.

    Args:
        action: Description of the action to confirm
        details: Additional details about the action

    Returns:
        A message indicating confirmation is pending
    """
    store = get_interaction_store()
    store.pending_confirmation = {
        "action": action,
        "details": details,
    }

    msg = f"PENDING_CONFIRMATION: {action}"
    if details:
        msg += f"\nDetails: {details}"
    return msg
