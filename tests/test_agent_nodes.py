"""Unit tests for agent.nodes module."""

from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph_code.agent.nodes import (
    get_tools,
    agent_node,
    tools_node,
    check_interaction_node,
    handle_interaction_response,
    should_continue,
)
from graph_code.agent.state import create_initial_state
from graph_code.config import Config
from graph_code.tools.interaction import get_interaction_store


class TestGetTools:
    """Tests for get_tools function."""

    def test_returns_list_of_tools(self):
        """Test that get_tools returns a list of tools."""
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 28

    def test_tools_have_names(self):
        """Test that all tools have names."""
        tools = get_tools()
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "read_file", "write_file", "edit_file", "bash", "search_files",
            "todo", "load_skill", "compact", "save_memory",
            "task_create", "task_update", "task_get", "task_list", "task_complete",
            "background_run", "background_check",
            "schedule_create", "schedule_list", "schedule_delete",
            "team_spawn", "send_message", "request_shutdown", "submit_plan_approval",
            "claim_task", "worktree_create", "worktree_enter", "worktree_run",
            "worktree_closeout",
        ]
        for name in expected_names:
            assert name in tool_names

    def test_tool_wrappers_delegate_to_runtime(self, tmp_path):
        """Model-visible StructuredTool wrappers should execute real runtime behavior."""
        (tmp_path / "sample.txt").write_text("hello from runtime\n")
        config = Config.for_tests(working_dir=tmp_path, model="mock")

        with patch("graph_code.agent.nodes.get_config", return_value=config):
            tools = {tool.name: tool for tool in get_tools()}
            read_result = tools["read_file"].invoke({"file_path": "sample.txt"})
            write_result = tools["write_file"].invoke(
                {"file_path": "created.txt", "content": "created"}
            )

        assert "hello from runtime" in read_result
        assert "Wrote file: created.txt" in write_result
        assert (tmp_path / "created.txt").read_text() == "created"


class TestAgentMessageSanitization:
    """Tests for removing invalid Unicode from model message history."""

    def test_agent_node_sanitizes_surrogates_in_history_before_llm_call(self):
        """Test that invalid surrogate characters are removed before API calls."""
        state = create_initial_state()
        state["messages"] = [AIMessage(content="bad \ud83d history")]

        with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            def invoke(messages):
                for message in messages:
                    if isinstance(message.content, str):
                        message.content.encode("utf-8")
                return AIMessage(content="ok")

            mock_llm.bind_tools.return_value.invoke.side_effect = invoke

            result = agent_node(state)

        assert result["final_response"] == "ok"
        assert state["messages"][0].content == "bad ? history"

    def test_agent_node_sanitizes_surrogates_in_model_response(self):
        """Test that invalid surrogate characters are not stored in responses."""
        state = create_initial_state()
        state["messages"] = [HumanMessage(content="hi")]

        with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(
                content="bad \ud83d response"
            )

            result = agent_node(state)

        assert result["final_response"] == "bad ? response"
        assert result["messages"][0].content == "bad ? response"


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

        result = tools_node(state)

        assert result["tool_calls"] == []
        assert result["iteration_count"] == 1

    def test_does_not_mutate_assistant_tool_messages_with_reasoning_content(self):
        """Test that tool execution does not add provider-specific message fields."""
        state = create_initial_state()
        assistant_message = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "_list_directory", "args": {"dir_path": "."}}],
        )
        state["messages"].append(assistant_message)
        state["tool_calls"] = assistant_message.tool_calls

        tools_node(state)

        assert "reasoning_content" not in assistant_message.additional_kwargs


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
