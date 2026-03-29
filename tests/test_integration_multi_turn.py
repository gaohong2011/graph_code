"""Integration tests for multi-turn conversation scenarios.

These tests verify that the agent can handle multiple user inputs
in sequence without state corruption or message mismatches.
"""

from unittest.mock import patch, MagicMock, call

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph_code.agent.graph import run_agent, resume_with_interaction
from graph_code.agent.state import create_initial_state


class TestMultiTurnConversation:
    """Tests for multi-turn conversation handling."""

    def test_second_command_clears_final_response(self):
        """Test that final_response is cleared before second command.

        This is a regression test for the bug where the second command
        would not execute because final_response was still set from
        the first command.
        """
        state = create_initial_state()
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            # First turn - returns final_response
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "First response"}}
            ]

            events1 = list(run_agent("first command", state, thread_id))
            assert state["final_response"] is None  # Should be cleared by run_agent
            assert len(events1) == 1
            assert events1[0]["final_response"] == "First response"

            # Second turn - should also work
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "Second response"}}
            ]

            events2 = list(run_agent("second command", state, thread_id))
            assert state["final_response"] is None
            assert len(events2) == 1
            assert events2[0]["final_response"] == "Second response"

    def test_multi_turn_preserves_message_history(self):
        """Test that message history is preserved across turns."""
        state = create_initial_state()
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            # First command
            list(run_agent("command 1", state, thread_id))
            assert len(state["messages"]) == 1
            assert state["messages"][0].content == "command 1"

            # Second command
            list(run_agent("command 2", state, thread_id))
            assert len(state["messages"]) == 2
            assert state["messages"][1].content == "command 2"

            # Third command
            list(run_agent("command 3", state, thread_id))
            assert len(state["messages"]) == 3
            assert state["messages"][2].content == "command 3"

    def test_clears_error_before_new_command(self):
        """Test that error state is cleared before processing new command."""
        state = create_initial_state()
        state["error"] = "Previous error"
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(run_agent("new command", state, thread_id))

            assert state["error"] is None

    def test_clears_pending_interaction_state(self):
        """Test that pending interaction flags are cleared."""
        state = create_initial_state()
        state["pending_question"] = True
        state["pending_confirmation"] = True
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(run_agent("command", state, thread_id))

            assert state["pending_question"] is False
            assert state["pending_confirmation"] is False


class TestToolCallMessageIntegrity:
    """Tests for tool call ID integrity across agent-tool interactions."""

    def test_tool_call_id_matches_between_call_and_result(self):
        """Test that tool call IDs match between assistant and tool messages.

        This is a regression test for the 'tool_call_id is not found' error
        that occurred when messages were manually duplicated.
        """
        state = create_initial_state()
        thread_id = "test-thread"

        tool_call = {
            "id": "call_abc123",
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }

        tool_result = ToolMessage(
            content="file content",
            tool_call_id="call_abc123"
        )

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            # Simulate agent -> tools -> agent flow
            mock_graph.stream.return_value = [
                {"agent": {"tool_calls": [tool_call], "messages": []}},
                {"tools": {"tool_results": [tool_result], "messages": [tool_result]}},
                {"agent": {"final_response": "Done", "messages": []}},
            ]

            events = list(run_agent("read test.py", state, thread_id))

            # Verify we got all events
            assert len(events) == 3

            # Verify tool call ID consistency
            agent_event = events[0]
            tools_event = events[1]

            call_id = agent_event["tool_calls"][0]["id"]
            result_id = tools_event["tool_results"][0].tool_call_id

            assert call_id == result_id, \
                f"Tool call ID mismatch: {call_id} != {result_id}"


class TestStateManagementInMain:
    """Tests that mirror the state management logic in main.py."""

    def test_main_style_state_update_does_not_duplicate_messages(self):
        """Test that state update logic from main.py doesn't duplicate messages.

        This replicates the bug where manual message merging in main.py
        caused message duplication and ID mismatches.
        """
        state = create_initial_state()
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            # First turn
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "Response 1"}}
            ]

            for event in run_agent("command 1", state, thread_id):
                # Simulate main.py state update (WITHOUT manual message merging)
                for key, value in event.items():
                    if key in state and key != "messages":
                        state[key] = value

            msg_count_after_first = len(state["messages"])

            # Second turn
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "Response 2"}}
            ]

            for event in run_agent("command 2", state, thread_id):
                for key, value in event.items():
                    if key in state and key != "messages":
                        state[key] = value

            msg_count_after_second = len(state["messages"])

            # Should have exactly 2 messages (one per turn)
            assert msg_count_after_second == 2, \
                f"Expected 2 messages, got {msg_count_after_second}"

    def test_old_bug_manual_message_merge_duplicates(self):
        """Demonstrate the old bug: manual message merging causes duplication.

        This test shows what happened with the old buggy code in main.py
        that manually merged messages.
        """
        state = create_initial_state()

        # Simulate what happened with manual merging
        msg1 = AIMessage(content="Response 1")
        msg2 = AIMessage(content="Response 2")

        # Old buggy approach: manual append
        state["messages"].append(msg1)

        # Then LangGraph's operator.add also adds the same message
        # (simulated by adding again)
        state["messages"].append(msg1)  # Duplicate!

        state["messages"].append(msg2)
        state["messages"].append(msg2)  # Duplicate!

        # This shows why we had 4 messages instead of 2
        assert len(state["messages"]) == 4  # Bug: duplicates


class TestResumeWithInteractionIntegration:
    """Integration tests for resume_with_interaction."""

    def test_resume_after_question_clears_state(self):
        """Test that resuming after a question clears pending state."""
        state = create_initial_state()
        state["pending_question"] = True
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "Thanks for the answer"}}
            ]

            events = list(resume_with_interaction(state, "my answer", thread_id))

            assert state["pending_question"] is False
            assert len(events) == 1

    def test_resume_preserves_previous_messages(self):
        """Test that resuming preserves messages from before the interaction."""
        state = create_initial_state()
        state["messages"] = [
            HumanMessage(content="original question"),
            AIMessage(content="I need to ask: what is your name?"),
        ]
        state["pending_question"] = True
        thread_id = "test-thread"

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(resume_with_interaction(state, "John", thread_id))

            # Should have original messages + user response
            assert len(state["messages"]) == 3
            assert state["messages"][-1].content == "John"
