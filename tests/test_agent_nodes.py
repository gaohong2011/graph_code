"""Unit tests for agent.nodes module."""

from unittest.mock import patch, MagicMock, call

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from graph_code.agent.nodes import (
    get_tools,
    agent_node,
    tools_node,
    check_interaction_node,
    handle_interaction_response,
    should_continue,
    _add_reasoning_content_to_messages,
)
from graph_code.agent.state import create_initial_state
from graph_code.tools.interaction import get_interaction_store


class TestGetTools:
    """Tests for get_tools function."""

    def test_returns_list_of_tools(self):
        """Test that get_tools returns a list of tools."""
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 10

    def test_tools_have_names(self):
        """Test that all tools have names."""
        tools = get_tools()
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "_read_file", "_write_file", "_list_directory", "_glob_search",
            "_grep_search", "_read_code_chunk",
            "_bash_command", "_python_execute",
            "_ask_user", "_confirm_action",
        ]
        for name in expected_names:
            assert name in tool_names


class TestAddReasoningContent:
    """Tests for _add_reasoning_content_to_messages function."""

    def test_adds_reasoning_content_to_ai_messages_with_tool_calls(self):
        """Test that reasoning_content is added to AIMessages with tool_calls."""
        msg = AIMessage(content="test", tool_calls=[{"id": "1", "name": "test_tool", "args": {}}])
        messages = [msg]

        _add_reasoning_content_to_messages(messages)

        assert msg.additional_kwargs.get("reasoning_content") == ""

    def test_does_not_add_to_messages_without_tool_calls(self):
        """Test that messages without tool_calls are not modified."""
        msg = AIMessage(content="test")
        messages = [msg]

        _add_reasoning_content_to_messages(messages)

        assert "reasoning_content" not in msg.additional_kwargs


class TestToolsNode:
    """Tests for tools_node function."""

    def test_empty_tool_calls_returns_empty(self):
        """Test that empty tool_calls returns empty dict."""
        state = create_initial_state()
        result = tools_node(state)
        assert result == {}

    def test_executes_tool_calls(self):
        """Test that tool calls are executed."""
        state = create_initial_state()
        state["tool_calls"] = [
            {"id": "call_1", "name": "_list_directory", "args": {"dir_path": "."}}
        ]

        with patch("graph_code.agent.nodes.ToolNode") as mock_tool_node_class:
            mock_tool_node = MagicMock()
            mock_tool_node_class.return_value = mock_tool_node
            mock_tool_node.invoke.return_value = {
                "messages": [ToolMessage(content="Result", tool_call_id="call_1")]
            }

            result = tools_node(state)

        assert result["tool_calls"] == []
        assert result["iteration_count"] == 1


class TestCheckInteractionNode:
    """Tests for check_interaction_node function."""

    def test_no_pending_interaction(self):
        """Test when there's no pending interaction."""
        get_interaction_store().clear()
        state = create_initial_state()

        result = check_interaction_node(state)

        assert result == {}

    def test_pending_question(self):
        """Test when there's a pending question."""
        store = get_interaction_store()
        store.clear()
        store.pending_question = "What is your name?"
        state = create_initial_state()

        result = check_interaction_node(state)

        assert result["pending_question"] is True
        assert result["pending_confirmation"] is False
        store.clear()

    def test_pending_confirmation(self):
        """Test when there's a pending confirmation."""
        store = get_interaction_store()
        store.clear()
        store.pending_confirmation = {"action": "Delete file"}
        state = create_initial_state()

        result = check_interaction_node(state)

        assert result["pending_question"] is False
        assert result["pending_confirmation"] is True
        store.clear()


class TestHandleInteractionResponse:
    """Tests for handle_interaction_response function."""

    def test_creates_human_message(self):
        """Test that user input is converted to HumanMessage."""
        state = create_initial_state()
        store = get_interaction_store()
        store.pending_question = "Question?"

        result = handle_interaction_response(state, "My answer")

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "My answer"

    def test_clears_pending_state(self):
        """Test that pending state is cleared."""
        state = create_initial_state()
        store = get_interaction_store()
        store.pending_question = "Question?"
        store.pending_confirmation = {"action": "Test"}

        result = handle_interaction_response(state, "answer")

        assert result["pending_question"] is False
        assert result["pending_confirmation"] is False
        assert store.pending_question is None
        assert store.pending_confirmation is None

    def test_includes_interaction_result(self):
        """Test that interaction_result is set."""
        state = create_initial_state()
        get_interaction_store().clear()

        result = handle_interaction_response(state, "user response")

        assert result["interaction_result"] == "user response"


class TestShouldContinue:
    """Tests for should_continue function."""

    def test_pause_on_pending_question(self):
        """Test that pending question causes pause."""
        state = create_initial_state()
        state["pending_question"] = True

        result = should_continue(state)

        assert result == "pause"

    def test_pause_on_pending_confirmation(self):
        """Test that pending confirmation causes pause."""
        state = create_initial_state()
        state["pending_confirmation"] = True

        result = should_continue(state)

        assert result == "pause"

    def test_end_on_final_response(self):
        """Test that final response causes end."""
        state = create_initial_state()
        state["final_response"] = "Done!"

        result = should_continue(state)

        assert result == "end"

    def test_end_on_error(self):
        """Test that error causes end."""
        state = create_initial_state()
        state["error"] = "Something went wrong"

        result = should_continue(state)

        assert result == "end"

    def test_end_on_iteration_limit(self):
        """Test that iteration limit causes end."""
        state = create_initial_state()
        state["iteration_count"] = 10

        with patch("graph_code.agent.nodes.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.max_tool_iterations = 10
            mock_get_config.return_value = mock_config

            result = should_continue(state)

        assert result == "end"

    def test_execute_tools_when_pending(self):
        """Test that pending tool calls cause execute_tools."""
        state = create_initial_state()
        state["tool_calls"] = [{"id": "call_1"}]

        result = should_continue(state)

        assert result == "execute_tools"

    def test_end_by_default(self):
        """Test that default is end."""
        state = create_initial_state()

        result = should_continue(state)

        assert result == "end"
