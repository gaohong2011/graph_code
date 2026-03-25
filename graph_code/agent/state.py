"""State definitions for Graph Code LangGraph agent."""

from typing import Annotated, List, Optional, TypedDict, Any
import operator

from langchain_core.messages import BaseMessage


class GraphCodeState(TypedDict):
    """State for the Graph Code agent.

    This state is passed between nodes in the LangGraph.
    """

    # Message history
    messages: Annotated[List[BaseMessage], operator.add]
    """List of messages in the conversation."""

    # Current task tracking
    current_task: Optional[str]
    """Description of the current task being worked on."""

    # Tool execution tracking
    tool_calls: List[dict]
    """List of pending tool calls to execute."""

    tool_results: List[dict]
    """Results from executed tool calls."""

    iteration_count: int
    """Number of tool execution iterations (to prevent infinite loops)."""

    # Human interaction state
    pending_confirmation: bool
    """Whether we're waiting for user confirmation."""

    pending_question: bool
    """Whether we're waiting for user to answer a question."""

    interaction_result: Optional[str]
    """Result from user interaction (answer or confirmation)."""

    # Final response
    final_response: Optional[str]
    """The final response to send to the user."""

    # Error tracking
    error: Optional[str]
    """Any error that occurred during execution."""


def create_initial_state() -> GraphCodeState:
    """Create an initial empty state."""
    return {
        "messages": [],
        "current_task": None,
        "tool_calls": [],
        "tool_results": [],
        "iteration_count": 0,
        "pending_confirmation": False,
        "pending_question": False,
        "interaction_result": None,
        "final_response": None,
        "error": None,
    }
