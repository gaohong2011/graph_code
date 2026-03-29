"""Integration tests using real ToolNode and actual tool execution.

These tests don't mock ToolNode to verify actual behavior with real tool calls.
This catches issues that mock-based tests miss, such as:
- ToolMessage ID mismatches
- Real error propagation
- Actual tool execution paths
"""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph_code.agent.nodes import tools_node, agent_node, get_tools
from graph_code.agent.state import create_initial_state
from graph_code.agent.graph import run_agent
from graph_code.config import reset_config


@pytest.fixture(autouse=True)
def setup_test_config(tmp_path, monkeypatch):
    """Setup test configuration before each test."""
    reset_config()
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.chdir(tmp_path)
    yield
    reset_config()


class TestRealToolExecution:
    """Tests using real tool execution (not mocked)."""

    def test_read_file_tool_creates_valid_tool_message(self, tmp_path, monkeypatch):
        """Test that read_file tool creates ToolMessage with valid tool_call_id."""
        # Setup temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Change working dir to temp path
        monkeypatch.chdir(tmp_path)

        # Create state with a real tool call
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "call_real_123",
            "name": "read_file",
            "args": {"file_path": "test.txt"},
        }]

        # Execute with real ToolNode
        result = tools_node(state)

        # Verify result
        assert "tool_results" in result
        assert len(result["tool_results"]) == 1

        tool_msg = result["tool_results"][0]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "call_real_123", \
            f"Expected 'call_real_123', got '{tool_msg.tool_call_id}'"
        assert "hello world" in tool_msg.content

    def test_multiple_real_tools_preserve_distinct_ids(self, tmp_path, monkeypatch):
        """Test multiple tool calls each get correct tool_call_id in responses."""
        monkeypatch.chdir(tmp_path)

        # Create test files
        (tmp_path / "a.txt").write_text("file A")
        (tmp_path / "b.txt").write_text("file B")

        state = create_initial_state()
        state["tool_calls"] = [
            {"id": "call_aaa", "name": "read_file", "args": {"file_path": "a.txt"}},
            {"id": "call_bbb", "name": "read_file", "args": {"file_path": "b.txt"}},
        ]

        result = tools_node(state)

        # Each result should have matching ID
        tool_msgs = result["tool_results"]
        assert len(tool_msgs) == 2

        ids = {msg.tool_call_id for msg in tool_msgs}
        assert ids == {"call_aaa", "call_bbb"}, f"Got IDs: {ids}"

        # Map content to ID
        id_content = {msg.tool_call_id: msg.content for msg in tool_msgs}
        assert "file A" in id_content["call_aaa"]
        assert "file B" in id_content["call_bbb"]

    def test_real_tool_error_still_has_valid_tool_call_id(self, tmp_path, monkeypatch):
        """Test that tool errors preserve tool_call_id."""
        monkeypatch.chdir(tmp_path)

        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "call_error_1",
            "name": "read_file",
            "args": {"file_path": "nonexistent.txt"},
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id == "call_error_1"
        assert "Error" in tool_msg.content or "not found" in tool_msg.content.lower()


class TestToolCallIdEdgeCases:
    """Tests for edge cases in tool_call_id handling."""

    def test_tool_call_with_none_id(self):
        """Test handling when tool_call id is None."""
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": None,  # Explicit None
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }]

        result = tools_node(state)

        # Should not crash and should have valid ID
        assert len(result["tool_results"]) == 1
        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id is not None
        assert tool_msg.tool_call_id != ""

    def test_tool_call_with_empty_string_id(self):
        """Test handling when tool_call id is empty string.

        This is the regression test for:
        'Invalid request: tool_call_id  is not found'
        """
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "",  # Empty string - the bug!
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        # Should use fallback ID, not empty string
        assert tool_msg.tool_call_id != "", \
            "tool_call_id should not be empty string - causes API error!"
        assert tool_msg.tool_call_id is not None

    def test_tool_call_missing_id_key(self):
        """Test handling when tool_call has no 'id' key at all."""
        state = create_initial_state()
        state["tool_calls"] = [{
            # No "id" key!
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }]

        result = tools_node(state)

        # Should handle gracefully
        assert len(result["tool_results"]) == 1
        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id is not None
        assert tool_msg.tool_call_id != ""

    def test_tool_call_with_whitespace_only_id(self):
        """Test handling when tool_call id is only whitespace."""
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "   ",  # Whitespace only
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        # Whitespace-only should also be treated as invalid
        assert tool_msg.tool_call_id.strip() != "", \
            "Whitespace-only tool_call_id should be handled"


class TestCompleteMessageFlow:
    """Tests verifying complete message flow from agent to tools and back."""

    def test_full_flow_with_real_tools(self, tmp_path, monkeypatch):
        """Test complete flow: agent -> tool_call -> tool_result -> agent.

        This verifies that:
        1. Agent produces AIMessage with tool_calls
        2. ToolNode executes and produces ToolMessages with matching IDs
        3. Messages can be assembled for next LLM call
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "hello.txt").write_text("Hello World")

        # Simulate state after agent_node produced tool_calls
        state = create_initial_state()
        state["messages"] = [
            HumanMessage(content="Read hello.txt"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "flow_test_123",
                    "name": "read_file",
                    "args": {"file_path": "hello.txt"},
                }]
            )
        ]
        state["tool_calls"] = [{
            "id": "flow_test_123",
            "name": "read_file",
            "args": {"file_path": "hello.txt"},
        }]

        # Execute tools
        tool_result = tools_node(state)

        # Update state with results (as done in real flow)
        state["messages"].extend(tool_result["messages"])
        state["tool_results"].extend(tool_result["tool_results"])

        # Verify complete message history
        assert len(state["messages"]) == 3  # human + assistant + tool

        # Verify assistant message has tool_calls
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert len(ai_msgs[0].tool_calls) == 1
        assert ai_msgs[0].tool_calls[0]["id"] == "flow_test_123"

        # Verify tool message matches
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "flow_test_123"

        # Verify ID consistency
        assert ai_msgs[0].tool_calls[0]["id"] == tool_msgs[0].tool_call_id


class TestBashToolRealExecution:
    """Tests for bash tool with real execution."""

    def test_bash_echo_preserves_tool_call_id(self):
        """Test bash command execution preserves tool_call_id."""
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "bash_call_456",
            "name": "bash_command",
            "args": {"command": "echo 'test output'"},
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id == "bash_call_456"
        assert "test output" in tool_msg.content

    def test_bash_error_preserves_tool_call_id(self):
        """Test failed bash command still preserves tool_call_id."""
        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "bash_error_789",
            "name": "bash_command",
            "args": {"command": "exit 1"},  # Command that fails
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id == "bash_error_789"


class TestGrepToolRealExecution:
    """Tests for grep tool with real file system."""

    def test_grep_finds_pattern_with_valid_tool_call_id(self, tmp_path, monkeypatch):
        """Test grep search returns results with correct tool_call_id."""
        monkeypatch.chdir(tmp_path)

        # Create a Python file with function definitions
        (tmp_path / "code.py").write_text("""
def foo():
    pass

def bar():
    pass
""")

        state = create_initial_state()
        state["tool_calls"] = [{
            "id": "grep_call_abc",
            "name": "grep_search",
            "args": {"pattern": "^def ", "glob": "*.py"},
        }]

        result = tools_node(state)

        tool_msg = result["tool_results"][0]
        assert tool_msg.tool_call_id == "grep_call_abc"
        assert "foo" in tool_msg.content or "bar" in tool_msg.content
