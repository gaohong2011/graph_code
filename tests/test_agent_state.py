"""Unit tests for agent.state module."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from graph_code.agent.state import GraphCodeState, create_initial_state


class TestGraphCodeState:
    """Tests for GraphCodeState TypedDict structure."""

    def test_state_can_hold_messages(self):
        """Test that state can hold messages."""
        state: GraphCodeState = {
            "messages": [HumanMessage(content="Hello")],
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
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)

    def test_state_can_hold_tool_calls(self):
        """Test that state can hold tool calls."""
        tool_call = {
            "id": "call_123",
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }
        state: GraphCodeState = {
            "messages": [],
            "current_task": "Testing",
            "tool_calls": [tool_call],
            "tool_results": [],
            "iteration_count": 1,
            "pending_confirmation": False,
            "pending_question": False,
            "interaction_result": None,
            "final_response": None,
            "error": None,
        }
        assert len(state["tool_calls"]) == 1
        assert state["tool_calls"][0]["name"] == "read_file"

    def test_state_can_hold_error(self):
        """Test that state can hold error information."""
        state: GraphCodeState = {
            "messages": [],
            "current_task": None,
            "tool_calls": [],
            "tool_results": [],
            "iteration_count": 0,
            "pending_confirmation": False,
            "pending_question": False,
            "interaction_result": None,
            "final_response": None,
            "error": "Something went wrong",
        }
        assert state["error"] == "Something went wrong"


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_initial_state_returns_empty_state(self):
        """Test that create_initial_state returns properly initialized state."""
        state = create_initial_state()

        assert state["messages"] == []
        assert state["current_task"] is None
        assert state["tool_calls"] == []
        assert state["tool_results"] == []
        assert state["iteration_count"] == 0
        assert state["pending_confirmation"] is False
        assert state["pending_question"] is False
        assert state["interaction_result"] is None
        assert state["final_response"] is None
        assert state["error"] is None

    def test_initial_state_is_independent(self):
        """Test that each call returns independent state objects."""
        state1 = create_initial_state()
        state2 = create_initial_state()

        # Should be different objects
        assert state1 is not state2

        # Modifying one should not affect the other
        state1["messages"].append(HumanMessage(content="Test"))
        assert len(state2["messages"]) == 0

    def test_initial_state_messages_is_list(self):
        """Test that messages field is a list."""
        state = create_initial_state()
        assert isinstance(state["messages"], list)
        # Should be able to append messages
        state["messages"].append(AIMessage(content="Response"))
        assert len(state["messages"]) == 1
