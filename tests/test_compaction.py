"""Context compaction behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph_code.agent.nodes import call_model, compact_check
from graph_code.agent.state import create_initial_state
from graph_code.config import Config
from graph_code.llm.protocol import validate_tool_message_protocol


def _compact_test_config(tmp_path, *, context_window_tokens: int = 1200) -> Config:
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.context_window_tokens = context_window_tokens
    config.auto_compact_ratio = 0.5
    config.micro_compact_ratio = 0.35
    config.compact_recent_messages = 1
    config.micro_compact_keep_tool_results = 1
    return config


def test_compact_check_uses_token_budget_not_message_count(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="large historical request " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path)

    result = compact_check(state, config=config)

    assert result["transition_reason"] == "summary_compact_complete"
    assert result["context_messages"]
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "large historical request" in context_text
    assert "x" * 1000 not in context_text
    assert "current request" in context_text
    assert result["compact_state"]["last_boundary_id"]


def test_micro_compact_preserves_tool_protocol_and_recent_result(tmp_path):
    old_call = {"id": "old-read", "name": "read_file", "args": {"file_path": "old.py"}}
    recent_call = {"id": "recent-read", "name": "read_file", "args": {"file_path": "new.py"}}
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="read old"),
        AIMessage(content="", tool_calls=[old_call]),
        ToolMessage(content="old output " + ("a" * 4000), tool_call_id="old-read"),
        HumanMessage(content="read recent"),
        AIMessage(content="", tool_calls=[recent_call]),
        ToolMessage(content="recent output", tool_call_id="recent-read"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=5000)

    result = compact_check(state, config=config)

    assert result["transition_reason"] == "micro_compact_complete"
    context_messages = result["context_messages"]
    assert validate_tool_message_protocol([SystemMessage(content="system"), *context_messages]) == []
    compacted_tool = next(
        message
        for message in context_messages
        if isinstance(message, ToolMessage) and message.tool_call_id == "old-read"
    )
    recent_tool = next(
        message
        for message in context_messages
        if isinstance(message, ToolMessage) and message.tool_call_id == "recent-read"
    )
    assert "[old tool result compacted]" in compacted_tool.content
    assert "a" * 1000 not in compacted_tool.content
    assert recent_tool.content == "recent output"


def test_summary_compact_keeps_protocol_safe_recent_suffix(tmp_path):
    old_call = {"id": "old-bash", "name": "bash", "args": {"command": "pytest"}}
    recent_call = {"id": "recent-read", "name": "read_file", "args": {"file_path": "README.md"}}
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old turn " + ("x" * 3000)),
        AIMessage(content="", tool_calls=[old_call]),
        ToolMessage(content="pytest output " + ("y" * 3000), tool_call_id="old-bash"),
        HumanMessage(content="current turn"),
        AIMessage(content="", tool_calls=[recent_call]),
        ToolMessage(content="README content", tool_call_id="recent-read"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.compact_recent_messages = 3

    result = compact_check(state, config=config)

    assert result["transition_reason"] == "summary_compact_complete"
    context_messages = result["context_messages"]
    assert validate_tool_message_protocol([SystemMessage(content="system"), *context_messages]) == []
    context_text = "\n".join(str(message.content) for message in context_messages)
    assert "Context compacted" in context_text
    assert "current turn" in context_text
    assert "README content" in context_text


def test_summary_compact_can_use_no_tools_model_summary(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old implementation detail " + ("x" * 4000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.llm_model = "real-model"
    config.llm_api_key = "test-key"
    config.compact_use_model_summary = True

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="LLM compact summary")

        result = compact_check(state, config=config)

    mock_llm.bind_tools.assert_not_called()
    mock_llm.invoke.assert_called_once()
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "LLM compact summary" in context_text


def test_call_model_uses_context_messages_when_available(tmp_path):
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="full transcript should not be sent")]
    state["context_messages"] = [HumanMessage(content="compacted context is sent")]
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    config.llm_api_key = "test-key"
    captured_messages = []

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        def invoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="ok")

        mock_llm.bind_tools.return_value.invoke.side_effect = invoke

        result = call_model(state, config=config)

    assert result["final_response"] == "ok"
    sent_text = "\n".join(str(message.content) for message in captured_messages)
    assert "compacted context is sent" in sent_text
    assert "full transcript should not be sent" not in sent_text


def test_call_model_appends_response_to_context_messages(tmp_path):
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="full historical transcript")]
    state["context_messages"] = [
        HumanMessage(content="[Context compacted: compact-test]"),
        HumanMessage(content="current compact context"),
    ]
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    config.llm_api_key = "test-key"

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="final answer")

        result = call_model(state, config=config)

    assert [message.content for message in result["context_messages"]][-2:] == [
        "current compact context",
        "final answer",
    ]
