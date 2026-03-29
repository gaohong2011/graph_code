"""Tests to reproduce and verify fix for tool_call_id bug.

The error was:
    Error code: 400 - {'error': {'message': 'Invalid request: tool_call_id  is not found', ...}}

Note the double space: 'tool_call_id  is not found'
This indicates the tool_call_id was empty string '' instead of a valid ID.
"""

from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph_code.agent.graph import run_agent
from graph_code.agent.state import create_initial_state


class TestToolCallIdPropagation:
    """Tests to verify tool_call_id is correctly propagated."""

    def test_tool_message_has_correct_tool_call_id(self):
        """Test that ToolMessage has the same tool_call_id as the tool call.

        This reproduces the bug where tool_call_id was empty or mismatched.
        """
        state = create_initial_state()

        # Simulate first turn: user asks for grep search
        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            # Simulate: agent -> tool_call -> tool_result -> final_response
            tool_call = {
                "id": "call_abc123",  # This is the ID that must be preserved
                "name": "grep_search",
                "args": {"pattern": "def ", "glob": "*.py"},
            }

            # Create a proper ToolMessage with the same ID
            tool_msg = ToolMessage(
                content="def foo():\ndef bar():",
                tool_call_id="call_abc123"  # Must match!
            )

            mock_graph.stream.return_value = [
                {"agent": {"tool_calls": [tool_call], "messages": []}},
                {"tools": {"tool_results": [tool_msg], "messages": [tool_msg], "iteration_count": 1}},
                {"agent": {"final_response": "Found 2 functions", "messages": []}},
            ]

            events = list(run_agent("search for def", state, "test"))

            # Find the tool result event
            tool_event = None
            for e in events:
                if "tool_results" in e:
                    tool_event = e
                    break

            assert tool_event is not None, "Should have a tool event"

            tool_msg = tool_event["tool_results"][0]
            assert tool_msg.tool_call_id == "call_abc123", \
                f"tool_call_id mismatch: expected 'call_abc123', got '{tool_msg.tool_call_id}'"

    def test_tool_call_id_not_empty_string(self):
        """Test that tool_call_id is never empty string.

        Empty string causes: 'tool_call_id  is not found' (note double space)
        """
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            tool_call = {
                "id": "call_xyz789",
                "name": "read_file",
                "args": {"file_path": "test.py"},
            }

            tool_msg = ToolMessage(
                content="file content",
                tool_call_id="call_xyz789"
            )

            mock_graph.stream.return_value = [
                {"agent": {"tool_calls": [tool_call]}},
                {"tools": {"tool_results": [tool_msg], "messages": [tool_msg]}},
            ]

            events = list(run_agent("read file", state, "test"))

            for e in events:
                if "tool_results" in e:
                    for msg in e["tool_results"]:
                        assert msg.tool_call_id != "", \
                            "tool_call_id should not be empty string! " \
                            "This causes 'tool_call_id  is not found' error"
                        assert msg.tool_call_id is not None, \
                            "tool_call_id should not be None"

    def test_multiple_tool_calls_have_distinct_ids(self):
        """Test that multiple tool calls in one turn have distinct IDs."""
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            tool_calls = [
                {"id": "call_1", "name": "read_file", "args": {"file_path": "a.py"}},
                {"id": "call_2", "name": "read_file", "args": {"file_path": "b.py"}},
            ]

            tool_msgs = [
                ToolMessage(content="content a", tool_call_id="call_1"),
                ToolMessage(content="content b", tool_call_id="call_2"),
            ]

            mock_graph.stream.return_value = [
                {"agent": {"tool_calls": tool_calls}},
                {"tools": {"tool_results": tool_msgs, "messages": tool_msgs}},
            ]

            events = list(run_agent("read two files", state, "test"))

            tool_event = [e for e in events if "tool_results" in e][0]
            ids = [msg.tool_call_id for msg in tool_event["tool_results"]]

            assert "call_1" in ids, "Should have tool result for call_1"
            assert "call_2" in ids, "Should have tool result for call_2"
            assert len(set(ids)) == 2, "Each tool call should have distinct ID"


class TestActualToolNodeBehavior:
    """Tests using actual ToolNode to verify real behavior."""

    def test_actual_tool_node_preserves_tool_call_id(self):
        """Test with real ToolNode to see actual behavior.

        This test doesn't mock ToolNode to ensure we're testing real behavior.
        """
        from graph_code.agent.nodes import tools_node, get_tools
        from langchain_core.messages import AIMessage

        # Create a real AIMessage with tool_calls
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "real_call_123",
                "name": "read_file",
                "args": {"file_path": "README.md"},
            }]
        )

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="read README")]
        state["tool_calls"] = ai_msg.tool_calls

        # Execute tools_node
        result = tools_node(state)

        # Check that tool results have correct IDs
        assert "tool_results" in result
        for msg in result["tool_results"]:
            if isinstance(msg, ToolMessage):
                assert msg.tool_call_id == "real_call_123", \
                    f"Expected 'real_call_123', got '{msg.tool_call_id}'"

    def test_tools_node_handles_missing_tool_call_id_gracefully(self):
        """Test behavior when tool_call_id is missing from tool_call dict."""
        from graph_code.agent.nodes import tools_node
        state = create_initial_state()

        # Tool call without 'id' key - this could cause issues
        state["tool_calls"] = [{
            "name": "read_file",
            "args": {"file_path": "test.py"},
            # Missing "id"!
        }]

        result = tools_node(state)

        # Should handle gracefully without crashing
        assert "tool_results" in result
        for msg in result["tool_results"]:
            if isinstance(msg, ToolMessage):
                # If id is missing, it should use "unknown" fallback
                assert msg.tool_call_id is not None


class TestMessageFlowIntegrity:
    """Tests verifying the complete message flow."""

    def test_assistant_message_with_tool_calls_in_history(self):
        """Verify that assistant message with tool_calls is in message history.

        The LLM needs to see both:
        1. The assistant message that made the tool call
        2. The tool response message

        If (1) is missing, the LLM API will complain about unknown tool_call_id.
        """
        from graph_code.agent.nodes import agent_node, tools_node

        # First, agent_node produces a tool call
        state = create_initial_state()
        state["messages"] = [HumanMessage(content="read README")]

        with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            # LLM responds with a tool call
            mock_response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_test_1",
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                }]
            )
            mock_llm.bind_tools.return_value.invoke.return_value = mock_response

            result = agent_node(state)

            # Check that the response is in messages
            assert len(result["messages"]) == 1
            assert result["messages"][0].tool_calls[0]["id"] == "call_test_1"

        # Now tools_node executes
        state["tool_calls"] = result["tool_calls"]
        state["messages"].extend(result["messages"])

        tool_result = tools_node(state)

        # Check the flow
        assert len(tool_result["messages"]) == 1
        tool_msg = tool_result["messages"][0]
        assert tool_msg.tool_call_id == "call_test_1"

        # The key: both messages should be in state
        state["messages"].extend(tool_result["messages"])
        assert len(state["messages"]) == 3  # human + assistant + tool

        # Verify the assistant message has tool_calls
        assistant_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0].tool_calls) == 1
        assert assistant_msgs[0].tool_calls[0]["id"] == "call_test_1"
