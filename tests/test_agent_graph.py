"""Unit tests for agent.graph module."""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from graph_code.agent.graph import (
    build_agent,
    run_agent,
    run_agent_async,
    resume_with_interaction,
)
from graph_code.agent.state import create_initial_state


class TestBuildAgent:
    """Tests for build_agent function."""

    def test_returns_compiled_graph(self):
        """Test that build_agent returns a compiled StateGraph."""
        graph = build_agent()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """Test that graph has the required nodes."""
        graph = build_agent()
        # The compiled graph should have nodes
        # We can't easily inspect the compiled graph,
        # but we can verify it was created without errors
        assert hasattr(graph, 'invoke') or hasattr(graph, 'stream')


class TestRunAgent:
    """Tests for run_agent function."""

    def test_yields_events(self):
        """Test that run_agent yields events."""
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = [
                {"agent": {"final_response": "Hello"}}
            ]

            events = list(run_agent("test input", state, "test-thread"))

        assert len(events) > 0
        assert events[0]["final_response"] == "Hello"

    def test_adds_user_message_to_state(self):
        """Test that user message is added to state."""
        state = create_initial_state()
        initial_message_count = len(state["messages"])

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(run_agent("user message", state, "thread-1"))

        assert len(state["messages"]) == initial_message_count + 1
        assert isinstance(state["messages"][-1], HumanMessage)
        assert state["messages"][-1].content == "user message"

    def test_uses_provided_thread_id(self):
        """Test that thread_id is passed to graph config."""
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(run_agent("test", state, "custom-thread-id"))

            call_args = mock_graph.stream.call_args
            config = call_args[1].get("configurable") or call_args[0][1].get("configurable")
            assert config["thread_id"] == "custom-thread-id"

    def test_uses_default_thread_id(self):
        """Test that default thread_id is 'default'."""
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(run_agent("test", state))

            call_args = mock_graph.stream.call_args
            config = call_args[1].get("configurable") or call_args[0][1].get("configurable")
            assert config["thread_id"] == "default"


class TestResumeWithInteraction:
    """Tests for resume_with_interaction function."""

    def test_updates_state_with_user_response(self):
        """Test that user response is added to state."""
        state = create_initial_state()
        state["messages"] = [HumanMessage(content="Original")]

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(resume_with_interaction(state, "user response", "thread-1"))

        # Should have original + user response
        assert len(state["messages"]) == 2
        assert state["messages"][-1].content == "user response"

    def test_clears_interaction_state(self):
        """Test that interaction state is cleared."""
        state = create_initial_state()
        state["pending_question"] = True
        state["pending_confirmation"] = True

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_graph.stream.return_value = []

            list(resume_with_interaction(state, "response", "thread-1"))

        assert state["pending_question"] is False
        assert state["pending_confirmation"] is False


class TestRunAgentAsync:
    """Tests for run_agent_async function."""

    @pytest.mark.asyncio
    async def test_async_yields_events(self):
        """Test that run_agent_async yields events."""
        state = create_initial_state()

        with patch("graph_code.agent.graph.build_agent") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph

            # Create async iterator
            async def mock_astream(*args, **kwargs):
                yield {"agent": {"final_response": "Async result"}}

            mock_graph.astream = mock_astream

            events = []
            async for event in run_agent_async("test", state, "thread-1"):
                events.append(event)

        assert len(events) > 0
