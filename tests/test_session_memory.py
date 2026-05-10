from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph_code.agent.session_memory.compact import load_session_memory_for_compact
from graph_code.agent.session_memory.prompt import DEFAULT_SESSION_MEMORY_TEMPLATE
from graph_code.agent.session_memory.state import should_update_session_memory
from graph_code.agent.session_memory.updater import maybe_update_session_memory
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_should_update_session_memory_respects_threshold(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="x" * 100)]

    assert should_update_session_memory(state, config) is True


def test_should_not_update_when_latest_assistant_has_tool_calls(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="x" * 100),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"file_path": "a.py"}}],
        ),
    ]

    assert should_update_session_memory(state, config) is False


def test_should_not_update_when_latest_assistant_tool_call_has_tool_result(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="x" * 100),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"file_path": "a.py"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
    ]
    state["pending_tool_calls"] = []
    state["tool_calls"] = []

    assert should_update_session_memory(state, config) is False


def test_should_update_when_tool_call_threshold_met_after_final_assistant(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10_000
    config.session_memory_tool_calls = 1
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="read a file"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"file_path": "a.py"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
        AIMessage(content="final answer"),
    ]

    assert should_update_session_memory(state, config) is True


def test_should_update_when_tool_call_growth_threshold_met_after_initialization(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_update_tokens = 10_000
    config.session_memory_tool_calls = 1
    state = create_initial_state()
    state["session_memory_state"].update(
        {
            "initialized": True,
            "tokens_at_last_update": 1000,
            "tool_calls_at_last_update": 1,
        }
    )
    state["messages"] = [
        HumanMessage(content="read files"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"file_path": "a.py"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
        HumanMessage(content="read another file"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-2", "name": "read_file", "args": {"file_path": "b.py"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-2"),
        AIMessage(content="final answer"),
    ]

    assert should_update_session_memory(state, config) is True


def test_maybe_update_session_memory_writes_mock_summary(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="implement feature")]

    update = maybe_update_session_memory(state, config)

    assert update["session_memory_state"]["initialized"] is True
    assert "session.md" in update["session_memory_state"]["path"]


def test_maybe_update_session_memory_records_tool_call_count(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10_000
    config.session_memory_tool_calls = 1
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="read a file"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"file_path": "a.py"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
        AIMessage(content="final answer"),
    ]

    update = maybe_update_session_memory(state, config)

    assert update["session_memory_state"]["tool_calls_at_last_update"] == 1


def test_maybe_update_session_memory_captures_memory_path_errors(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    config.session_memory_init_tokens = 10
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="x" * 100)]

    with patch(
        "graph_code.agent.session_memory.updater.memory_paths_for_project",
        side_effect=OSError("boom"),
    ):
        update = maybe_update_session_memory(state, config)

    assert update["session_memory_state"]["last_error"] == "OSError: boom"


def test_load_session_memory_for_compact_ignores_template(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.session_memory_enabled = True
    from graph_code.agent.memory.paths import memory_paths_for_project

    paths = memory_paths_for_project(config)
    paths.session_memory_dir.mkdir(parents=True)
    paths.session_memory_file.write_text(DEFAULT_SESSION_MEMORY_TEMPLATE, encoding="utf-8")

    assert load_session_memory_for_compact(config) is None
