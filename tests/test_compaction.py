"""Context compaction behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph_code.agent.graph import run_agent
from graph_code.agent.nodes import (
    append_tool_results,
    build_prompt,
    call_model,
    compact_check,
    execute_tools,
    recovery_handler,
)
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


def test_micro_compact_skips_non_compactable_tools(tmp_path):
    old_call = {"id": "send-1", "name": "send_message", "args": {"content": "large"}}
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="send message"),
        AIMessage(content="", tool_calls=[old_call]),
        ToolMessage(content="message output " + ("a" * 4000), tool_call_id="send-1"),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=5000)

    result = compact_check(state, config=config)

    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "[old tool result compacted]" not in context_text
    assert "a" * 1000 in context_text


def test_time_based_micro_compact_clears_old_tool_results_below_token_threshold(tmp_path):
    old_call = {"id": "old-read", "name": "read_file", "args": {"file_path": "old.py"}}
    recent_call = {"id": "recent-read", "name": "read_file", "args": {"file_path": "new.py"}}
    state = create_initial_state()
    state["turn_count"] = 10
    state["compact_state"]["last_main_loop_assistant_turn"] = 1
    state["messages"] = [
        HumanMessage(content="read old"),
        AIMessage(content="", tool_calls=[old_call]),
        ToolMessage(content="old output " + ("a" * 800), tool_call_id="old-read"),
        HumanMessage(content="read recent"),
        AIMessage(content="", tool_calls=[recent_call]),
        ToolMessage(content="recent output", tool_call_id="recent-read"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=20000)
    config.time_based_microcompact_turn_gap = 5

    result = compact_check(state, config=config)

    assert result["transition_reason"] == "micro_compact_complete"
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "[old tool result compacted]" in context_text
    assert "recent output" in context_text


def test_time_based_micro_compact_records_turn_to_prevent_immediate_repeat(tmp_path):
    old_call = {"id": "old-read", "name": "read_file", "args": {"file_path": "old.py"}}
    recent_call = {"id": "recent-read", "name": "read_file", "args": {"file_path": "recent.py"}}
    state = create_initial_state()
    state["turn_count"] = 10
    state["compact_state"]["last_main_loop_assistant_turn"] = 1
    state["messages"] = [
        HumanMessage(content="read old"),
        AIMessage(content="", tool_calls=[old_call]),
        ToolMessage(content="old output " + ("a" * 800), tool_call_id="old-read"),
        HumanMessage(content="read recent"),
        AIMessage(content="", tool_calls=[recent_call]),
        ToolMessage(content="recent output", tool_call_id="recent-read"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=20000)
    config.time_based_microcompact_turn_gap = 5

    first = compact_check(state, config=config)
    state["compact_state"] = first["compact_state"]
    state["turn_count"] = 12
    state["messages"].append(HumanMessage(content="new small request"))

    second = compact_check(state, config=config)

    assert first["transition_reason"] == "micro_compact_complete"
    assert first["compact_state"]["last_main_loop_assistant_turn"] == 10
    assert second["transition_reason"] == "compact_not_needed"


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


def test_summary_compact_places_rehydration_before_recent_tool_group(tmp_path):
    recent_call = {"id": "recent-read", "name": "read_file", "args": {"file_path": "README.md"}}
    state = create_initial_state()
    state["current_task_id"] = "task-123"
    state["messages"] = [
        HumanMessage(content="old turn " + ("x" * 3000)),
        HumanMessage(content="current turn"),
        AIMessage(content="", tool_calls=[recent_call]),
        ToolMessage(content="README content", tool_call_id="recent-read"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.compact_recent_messages = 2

    result = compact_check(state, config=config)

    context_messages = result["context_messages"]
    assert isinstance(context_messages[-2], AIMessage)
    assert isinstance(context_messages[-1], ToolMessage)
    assert "task-123" in "\n".join(str(message.content) for message in context_messages[:-2])


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


def test_model_summary_prompt_uses_required_sections_and_strips_analysis(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old implementation detail " + ("x" * 4000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.llm_model = "real-model"
    config.llm_api_key = "test-key"
    config.compact_use_model_summary = True
    captured_prompt = []

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        def invoke(messages):
            captured_prompt.extend(messages)
            return AIMessage(
                content=(
                    "<analysis>scratchpad should be removed</analysis>"
                    "<summary>Primary Request and Intent: continue current request\n"
                    "Current Work: implementation\n"
                    "Optional Next Step: run tests</summary>"
                )
            )

        mock_llm.invoke.side_effect = invoke

        result = compact_check(state, config=config)

    prompt_text = "\n".join(str(message.content) for message in captured_prompt)
    for section in [
        "Primary Request and Intent",
        "Key Technical Concepts",
        "Files and Code Sections",
        "Errors and fixes",
        "Problem Solving",
        "All user messages",
        "Pending Tasks",
        "Current Work",
        "Optional Next Step",
    ]:
        assert section in prompt_text
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "scratchpad should be removed" not in context_text
    assert "Primary Request and Intent: continue current request" in context_text


def test_model_summary_prompt_too_long_retries_with_short_prompt(tmp_path):
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
        mock_llm.invoke.side_effect = [
            Exception("prompt too long"),
            AIMessage(content="<summary>Retried compact summary</summary>"),
        ]

        result = compact_check(state, config=config)

    assert mock_llm.invoke.call_count == 2
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "Retried compact summary" in context_text


def test_model_summary_circuit_breaker_skips_after_failures(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old implementation detail " + ("x" * 4000)),
        HumanMessage(content="current request"),
    ]
    state["compact_state"]["consecutive_failures"] = 3
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.llm_model = "real-model"
    config.llm_api_key = "test-key"
    config.compact_use_model_summary = True

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        result = compact_check(state, config=config)

    mock_get_llm.assert_not_called()
    assert result["compact_state"]["consecutive_failures"] == 3
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "Context compacted" in context_text


def test_compact_warning_state_records_high_context_without_compaction(tmp_path):
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="medium context " + ("x" * 3000))]
    config = _compact_test_config(tmp_path, context_window_tokens=10000)
    config.auto_compact_ratio = 0.9
    config.micro_compact_ratio = 0.8
    config.compact_warning_ratio = 0.1
    config.compact_message_count_threshold = 100

    result = compact_check(state, config=config)

    assert result["transition_reason"] == "compact_not_needed"
    warning = result["compact_state"]["warning_state"]
    assert warning["warning"] is True
    assert warning["estimated_tokens"] > warning["warning_threshold"]


def test_summary_compact_rehydrates_runtime_context(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    state["current_task_id"] = "task-123"
    state["planning_state"] = {"status": "approved", "approved": True}
    state["loaded_skills"] = {"python": {"path": ".agent/skills/python/SKILL.md"}}
    state["worktree_context"] = {"current": "wt-1", "registry": {"wt-1": {"path": "worktrees/wt-1"}}}
    state["mcp_connection_state"] = {"fs": {"status": "connected", "tools": ["read"]}}
    config = _compact_test_config(tmp_path, context_window_tokens=1000)

    result = compact_check(state, config=config)

    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    for expected in ["task-123", "approved", "python", "wt-1", "fs", "connected"]:
        assert expected in context_text


def test_execute_tools_records_recent_file_context(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["pending_tool_calls"] = [
        {"id": "read-a", "name": "read_file", "args": {"file_path": "a.py"}}
    ]

    result = execute_tools(state, config=config)

    recent = result["file_context_state"]["recent_files"]
    assert recent[-1]["path"] == "a.py"
    assert recent[-1]["tool"] == "read_file"
    assert "print" in recent[-1]["preview"]


def test_summary_compact_rehydrates_recent_file_context(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    state["file_context_state"]["recent_files"] = [
        {"path": "a.py", "tool": "read_file", "preview": "def a(): pass", "turn": 1}
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)

    result = compact_check(state, config=config)

    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "Recent file context" in context_text
    assert "a.py" in context_text
    assert "def a" in context_text


def test_summary_compact_writes_transcript_and_records_path(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)

    result = compact_check(state, config=config)

    transcript_path = result["compact_state"]["transcript_path"]
    transcript = tmp_path / transcript_path
    assert transcript.exists()
    lines = transcript.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "human"
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert transcript_path in context_text


def test_summary_compact_runs_pre_and_post_hooks(tmp_path):
    hook_dir = tmp_path / ".agent" / "hooks"
    hook_dir.mkdir(parents=True)
    (hook_dir / "pre_compact.py").write_text("print('pre hook instruction')\n", encoding="utf-8")
    (hook_dir / "post_compact.py").write_text("print('post hook context')\n", encoding="utf-8")
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=1000)

    result = compact_check(state, config=config)

    summary = result["compact_state"]["summaries"][-1]
    assert "pre hook instruction" in summary["pre_compact_hooks"][0]["content"]
    context_text = "\n".join(str(message.content) for message in result["context_messages"])
    assert "post hook context" in context_text


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


def test_build_prompt_compacts_long_context_before_model_call(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="historical context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path)

    result = build_prompt(state, config=config)

    assert result["transition_reason"] == "summary_compact_complete"
    assert result["context_messages"]
    assert result["compact_state"]["mode"] == "summary"


def test_summary_compact_invalidates_prompt_cache(tmp_path):
    state = create_initial_state()
    state["prompt_state"]["cache"] = {"memory": "old"}
    state["messages"] = [
        HumanMessage(content="historical context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path)

    result = build_prompt(state, config=config)

    assert result["transition_reason"] == "summary_compact_complete"
    assert result["system_prompt"]
    assert result["prompt_state"]["cache"].get("memory") != "old"
    assert result["prompt_state"]["invalidated"] is False


def test_reactive_context_compact_rebuilds_system_prompt(tmp_path):
    state = create_initial_state()
    state["error"] = "Error code: 400 - context length exceeded"
    state["system_prompt"] = "OLD SYSTEM PROMPT"
    state["prompt_state"]["cache"] = {"memory": "old"}
    state["messages"] = [
        HumanMessage(content="historical context " + ("x" * 6000)),
        HumanMessage(content="current request"),
    ]
    config = _compact_test_config(tmp_path)

    result = recovery_handler(state, config=config)

    assert result["transition_reason"] == "context_compact_retry"
    assert result["system_prompt"]
    assert result["system_prompt"] != "OLD SYSTEM PROMPT"
    assert result["prompt_state"]["cache"].get("memory") != "old"
    assert result["prompt_state"]["invalidated"] is False


def test_manual_compact_request_survives_tool_result_append(tmp_path):
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="old context " + ("x" * 6000)),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "compact-call", "name": "compact", "args": {"summary": "manual note"}}
            ],
        ),
    ]
    state["tool_results"] = [
        {
            "ok": True,
            "content": '{"mode": "manual", "summary": "manual note"}',
            "is_error": False,
            "attachments": [],
            "metadata": {"tool_name": "compact"},
            "tool_call_id": "compact-call",
        }
    ]
    config = _compact_test_config(tmp_path, context_window_tokens=10000)

    appended = append_tool_results(state)
    state.update({key: value for key, value in appended.items() if key != "messages"})
    state["messages"].extend(appended["messages"])
    compacted = compact_check(state, config=config)

    assert compacted["transition_reason"] == "summary_compact_complete"
    summary = compacted["compact_state"]["summaries"][-1]
    assert summary["manual_summary"] == "manual note"


def test_context_too_long_error_triggers_reactive_compact_retry(tmp_path):
    config = _compact_test_config(tmp_path, context_window_tokens=1000)
    config.llm_model = "real-model"
    config.llm_api_key = "test-key"
    config.compact_use_model_summary = False
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="large request " + ("x" * 5000))]

    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        Exception("Error code: 400 - context length exceeded"),
        AIMessage(content="Recovered after reactive compact"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model

    with patch("graph_code.agent.nodes.get_llm", return_value=llm):
        events = list(run_agent("continue", state, "context-retry", config=config))

    final_responses = [event["final_response"] for event in events if event.get("final_response")]
    assert final_responses[-1] == "Recovered after reactive compact"
    assert bound_model.invoke.call_count == 2
    assert state["recovery_state"]["context_retry_budget"] == 0
    assert state["compact_state"]["mode"] == "summary"
