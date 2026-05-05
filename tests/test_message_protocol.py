"""Protocol checks for OpenAI-compatible tool-call message history."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph_code.agent.nodes import call_model
from graph_code.agent.state import create_initial_state
from graph_code.config import Config
from graph_code.llm.protocol import validate_tool_message_protocol


def test_validate_tool_message_protocol_accepts_adjacent_tool_results():
    tool_calls = [
        {"id": "call_1", "name": "read_file", "args": {"file_path": "a.py"}},
        {"id": "call_2", "name": "read_file", "args": {"file_path": "b.py"}},
    ]

    errors = validate_tool_message_protocol(
        [
            SystemMessage(content="system"),
            HumanMessage(content="read files"),
            AIMessage(content="", tool_calls=tool_calls),
            ToolMessage(content="a", tool_call_id="call_1"),
            ToolMessage(content="b", tool_call_id="call_2"),
            AIMessage(content="done"),
        ]
    )

    assert errors == []


def test_validate_tool_message_protocol_rejects_missing_tool_result():
    errors = validate_tool_message_protocol(
        [
            HumanMessage(content="run command"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "bash:2", "name": "bash", "args": {"command": "python hello.py"}}
                ],
            ),
            HumanMessage(content="next turn"),
        ]
    )

    assert errors
    assert "bash:2" in errors[0]


def test_call_model_returns_local_protocol_error_before_provider_call(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="run command"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "bash:2", "name": "bash", "args": {"command": "python hello.py"}}
            ],
        ),
        HumanMessage(content="next turn"),
    ]
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    config.llm_api_key = "test-key"

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()

        result = call_model(state, config=config)

    mock_get_llm.assert_not_called()
    assert result["transition_reason"] == "message_protocol_error"
    assert "bash:2" in result["error"]
