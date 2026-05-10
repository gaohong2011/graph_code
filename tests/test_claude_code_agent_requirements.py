"""Requirement tests for the full LangGraph coding agent."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from graph_code.agent.graph import build_agent, run_agent
from graph_code.agent.persistence import create_checkpointer, create_store
from graph_code.agent.state import AgentState, create_initial_state
from graph_code.agent.nodes import (
    append_tool_results,
    compact_check,
    execute_tools,
    human_permission_interrupt,
    permission_gate,
)
from graph_code.agent.subagents import run_subagent
from graph_code.config import Config
from graph_code.entities.background import background_check, background_run
from graph_code.entities.schedules import schedule_create, schedule_delete, schedule_list
from graph_code.entities.tasks import task_create, task_complete, task_get, task_update
from graph_code.entities.tasks import claim_task as persistent_claim_task
from graph_code.entities.teams import send_message, team_spawn
from graph_code.entities.worktrees import (
    WorktreeDirtyError,
    worktree_closeout,
    worktree_create,
    worktree_enter,
)
from graph_code.mcp.client import MCPClientRegistry, SDKMCPClient
from graph_code.tools.permissions import PermissionMode
from graph_code.tools.runtime import ToolExecutionRuntime
from graph_code.tools.schema import ToolResultEnvelope


def test_initial_state_contains_required_custom_fields():
    state: AgentState = create_initial_state()

    for key in [
        "messages",
        "turn_count",
        "transition_reason",
        "pending_tool_calls",
        "pending_permission_request",
        "tool_results",
        "planning_state",
        "compact_state",
        "recovery_state",
        "loaded_skills",
        "notifications",
        "runtime_tasks",
        "current_task_id",
        "teammate_identity",
        "worktree_context",
        "mcp_connection_state",
    ]:
        assert key in state

    assert state["turn_count"] == 0
    assert state["pending_tool_calls"] == []
    assert state["notifications"] == []

    for key in [
        "system_prompt",
        "prompt_state",
        "memory_state",
        "session_memory_state",
        "file_context_state",
    ]:
        assert key in state

    assert state["system_prompt"] == ""
    assert state["prompt_state"]["cache"] == {}
    assert state["memory_state"]["surfaced_memories"] == []
    assert state["session_memory_state"]["initialized"] is False
    assert state["file_context_state"]["recent_files"] == []


def test_persistence_factories_return_memory_backends(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.checkpoint_backend = "memory"
    config.store_backend = "memory"

    assert create_checkpointer(config).__class__.__name__ == "InMemorySaver"
    assert create_store(config).__class__.__name__ == "InMemoryStore"


def test_persistence_factories_raise_helpful_error_for_missing_sqlite(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.checkpoint_backend = "sqlite"
    config.checkpoint_uri = str(tmp_path / "checkpoints.sqlite")

    with pytest.raises(RuntimeError, match="langgraph-checkpoint-sqlite"):
        create_checkpointer(config)


def test_graph_interrupts_for_dangerous_bash_and_resumes_with_denial(tmp_path):
    graph = build_agent(config=Config.for_tests(working_dir=tmp_path, model="mock"))
    thread = {"configurable": {"thread_id": "perm-deny"}}
    tool_call = {
        "id": "call_1",
        "name": "bash",
        "args": {"command": "sudo rm -rf build"},
    }
    state = create_initial_state(permission_mode=PermissionMode.DEFAULT.value)
    state["messages"] = [AIMessage(content="", tool_calls=[tool_call])]
    state["pending_tool_calls"] = [tool_call]

    interrupted = graph.invoke(state, thread, interrupt_before=())

    assert "__interrupt__" in interrupted
    request = interrupted["__interrupt__"][0].value
    assert request["tool_name"] == "bash"
    assert request["risk"] == "dangerous_command"

    resumed = graph.invoke(Command(resume={"approved": False, "reason": "no"}), thread)

    tool_messages = [m for m in resumed["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages
    assert tool_messages[-1].tool_call_id == "call_1"
    assert "Permission denied" in tool_messages[-1].content


def test_permission_interrupt_preserves_prior_allowed_tool_calls(tmp_path):
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    graph = build_agent(config=Config.for_tests(working_dir=tmp_path, model="mock"))
    thread = {"configurable": {"thread_id": "perm-mixed"}}
    read_call = {
        "id": "read_1",
        "name": "read_file",
        "args": {"file_path": "hello.txt"},
    }
    write_call = {
        "id": "write_1",
        "name": "write_file",
        "args": {"file_path": "out.txt", "content": "ok"},
    }
    state = create_initial_state(permission_mode=PermissionMode.DEFAULT.value)
    state["messages"] = [AIMessage(content="", tool_calls=[read_call, write_call])]
    state["pending_tool_calls"] = [read_call, write_call]

    interrupted = graph.invoke(state, thread, interrupt_before=())

    assert "__interrupt__" in interrupted

    resumed = graph.invoke(Command(resume={"approved": True}), thread)

    tool_messages = [m for m in resumed["messages"] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages[-2:]} == {"read_1", "write_1"}


def test_multiple_side_effect_tool_calls_require_separate_approvals(tmp_path):
    graph = build_agent(config=Config.for_tests(working_dir=tmp_path, model="mock"))
    thread = {"configurable": {"thread_id": "perm-multiple"}}
    write_call = {
        "id": "write_1",
        "name": "write_file",
        "args": {"file_path": "out.txt", "content": "ok"},
    }
    bash_call = {
        "id": "bash_1",
        "name": "bash",
        "args": {"command": "touch should_not_exist"},
    }
    state = create_initial_state(permission_mode=PermissionMode.DEFAULT.value)
    state["messages"] = [AIMessage(content="", tool_calls=[write_call, bash_call])]
    state["pending_tool_calls"] = [write_call, bash_call]

    first = graph.invoke(state, thread, interrupt_before=())

    assert first["__interrupt__"][0].value["tool_name"] == "write_file"

    second = graph.invoke(Command(resume={"approved": True}), thread)

    assert "__interrupt__" in second
    assert second["__interrupt__"][0].value["tool_name"] == "bash"

    final = graph.invoke(Command(resume={"approved": False, "reason": "no bash"}), thread)

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok"
    assert not (tmp_path / "should_not_exist").exists()
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages[-2:]} == {"write_1", "bash_1"}
    assert "Permission denied" in tool_messages[-1].content


def test_permission_gate_returns_denied_tool_result_without_crashing(tmp_path):
    state = create_initial_state(permission_mode=PermissionMode.AUTO.value)
    state["pending_tool_calls"] = [
        {"id": "call_1", "name": "bash", "args": {"command": "rm -rf /"}}
    ]

    result = permission_gate(state)

    assert result["pending_tool_calls"] == []
    assert result["tool_results"][0]["ok"] is False
    assert result["tool_results"][0]["is_error"] is True
    assert "blocked" in result["tool_results"][0]["content"].lower()


def test_tool_runtime_persists_large_outputs_and_preserves_order(tmp_path):
    runtime = ToolExecutionRuntime(working_dir=tmp_path, output_limit=32)
    calls = [
        {"id": "one", "name": "read_file", "args": {"file_path": "missing.txt"}},
        {"id": "two", "name": "bash", "args": {"command": "printf 'abcdefghijklmnopqrstuvwxyz0123456789'"}},
    ]

    results = runtime.execute(calls, permission_mode=PermissionMode.AUTO)

    assert [result.tool_call_id for result in results] == ["one", "two"]
    assert results[1].metadata["persisted_output"] is not None
    persisted = tmp_path / results[1].metadata["persisted_output"]
    assert persisted.exists()
    assert persisted.read_text().endswith("abcdefghijklmnopqrstuvwxyz0123456789")


def test_tool_runtime_preserves_read_write_execution_order(tmp_path):
    (tmp_path / "state.txt").write_text("old", encoding="utf-8")
    runtime = ToolExecutionRuntime(working_dir=tmp_path)
    calls = [
        {"id": "read_before", "name": "read_file", "args": {"file_path": "state.txt"}},
        {"id": "write", "name": "write_file", "args": {"file_path": "state.txt", "content": "new"}},
        {"id": "read_after", "name": "read_file", "args": {"file_path": "state.txt"}},
    ]

    results = runtime.execute(calls, permission_mode=PermissionMode.AUTO)

    assert [result.tool_call_id for result in results] == ["read_before", "write", "read_after"]
    assert "old" in results[0].content
    assert "new" in results[2].content


def test_append_tool_results_writes_tool_messages_with_envelopes():
    state = create_initial_state()
    state["tool_results"] = [
        ToolResultEnvelope(
            ok=True,
            content="hello",
            is_error=False,
            tool_call_id="call_1",
            metadata={"tool_name": "read_file"},
        ).model_dump()
    ]

    update = append_tool_results(state)

    assert isinstance(update["messages"][0], ToolMessage)
    assert update["messages"][0].tool_call_id == "call_1"
    payload = json.loads(update["messages"][0].content)
    assert payload["ok"] is True
    assert payload["content"] == "hello"


def test_task_completion_unlocks_dependent_tasks(tmp_path):
    first = task_create(tmp_path, subject="first", description="one")
    second = task_create(
        tmp_path,
        subject="second",
        description="two",
        blocked_by=[first.id],
    )

    assert task_get(tmp_path, second.id).status == "blocked"

    task_complete(tmp_path, first.id)

    assert task_get(tmp_path, second.id).status == "pending"


def test_background_completion_is_drained_as_notification(tmp_path):
    record = background_run(tmp_path, command="python -c \"print('done')\"")

    notification = background_check(tmp_path, record.id)

    assert notification is not None
    assert notification["type"] == "runtime_task_completed"
    assert Path(notification["output_path"]).exists()


def test_scheduler_due_records_only_enqueue_notifications(tmp_path):
    created = schedule_create(
        tmp_path,
        cron="* * * * *",
        prompt="run health check",
        recurring=False,
        durable=True,
    )

    notifications = schedule_list(tmp_path, now=created.created_at)

    assert notifications[0]["type"] == "schedule_due"
    assert notifications[0]["schedule_id"] == created.id
    assert schedule_delete(tmp_path, created.id).id == created.id


def test_worktree_closeout_refuses_dirty_remove(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    (base / ".git").mkdir()
    registry_root = tmp_path / "registry"

    record = worktree_create(registry_root, task_id="task-1", base_path=base)
    entered = worktree_enter(registry_root, record.id)
    (Path(entered.path) / "changed.txt").write_text("dirty")

    with pytest.raises(WorktreeDirtyError):
        worktree_closeout(registry_root, record.id, mode="remove")


def test_mock_mcp_tool_routes_through_registry(tmp_path):
    registry = MCPClientRegistry(tmp_path)
    registry.load_manifest(
        {
            "servers": {
                "mock": {
                    "transport": "mock",
                    "tools": {
                        "echo": {"response": "pong"},
                    },
                }
            }
        }
    )

    result = registry.call_tool("mcp__mock__echo", {"value": "ping"})

    assert result.ok is True
    assert result.content == "pong"
    assert registry.connection_state["mock"]["status"] == "connected"


def test_mcp_registry_uses_sdk_client_for_real_transports(monkeypatch, tmp_path):
    connected = []

    class FakeSDKClient:
        def __init__(self, name, config):
            self.name = name
            self.config = config

        def connect(self):
            connected.append((self.name, self.config["transport"]))

        def list_tools(self):
            return ["echo"]

        def call_tool(self, tool, args):
            return ToolResultEnvelope.success("ok")

    monkeypatch.setattr("graph_code.mcp.client.SDKMCPClient", FakeSDKClient)
    registry = MCPClientRegistry(tmp_path)

    registry.load_manifest(
        {
            "servers": {
                "real": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                }
            }
        }
    )

    assert connected == [("real", "stdio")]
    assert registry.list_tools() == ["mcp__real__echo"]
    assert registry.connection_state["real"]["status"] == "connected"


def test_sdk_mcp_client_reports_needs_auth_without_token(monkeypatch):
    config = {
        "transport": "streamable-http",
        "url": "https://example.test/mcp",
        "auth": {"type": "bearer", "token_env": "MISSING_TOKEN"},
    }
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    client = SDKMCPClient("private", config)

    with pytest.raises(RuntimeError, match="needs-auth"):
        client.connect()


def test_worktree_create_uses_git_worktree_add_for_real_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    registry_root = tmp_path / "registry"
    record = worktree_create(registry_root, task_id="task-actual", base_path=repo)

    assert (Path(record.path) / ".git").exists()
    assert any(event["event"] == "git_worktree_add" for event in record.event_log)


def test_todo_rejects_multiple_in_progress_items(tmp_path):
    runtime = ToolExecutionRuntime(working_dir=tmp_path)

    result = runtime.todo(
        action="set",
        items=[
            {"id": "1", "content": "first", "status": "in_progress"},
            {"id": "2", "content": "second", "status": "in_progress"},
        ],
    )

    assert "Error" in result
    assert "one in_progress" in result


def test_compact_summary_preserves_required_fields():
    state = create_initial_state()
    state["messages"] = [HumanMessage(content=f"message {index}") for index in range(45)]

    result = compact_check(state)

    summary = result["compact_state"]["summaries"][0]
    for key in ["current_goal", "completed_actions", "key_files", "key_decisions", "next_step"]:
        assert key in summary


def test_teammate_inbox_uses_request_records(tmp_path):
    teammate = team_spawn(tmp_path, name="worker-a", role="implementer")
    request = send_message(tmp_path, teammate.id, "please review", request_id="req-fixed")

    assert request.id == "req-fixed"
    reloaded = (tmp_path / ".agent" / "teams" / f"{teammate.id}.json").read_text()
    assert "req-fixed" in reloaded


def test_claim_task_uses_lock_and_event_log(tmp_path):
    task = task_create(tmp_path, subject="claimable", description="")

    claimed = persistent_claim_task(tmp_path, task.id, owner="agent-a")

    assert claimed.owner == "agent-a"
    assert claimed.event_log[-1]["event"] == "claimed"
    with pytest.raises(RuntimeError):
        persistent_claim_task(tmp_path, task.id, owner="agent-b")


def test_subagent_returns_summary_from_subgraph(tmp_path):
    summary = run_subagent(
        "inspect README",
        config=Config.for_tests(working_dir=tmp_path, model="mock"),
    )

    assert summary["status"] == "completed"
    assert "inspect README" in summary["summary"]


def test_run_agent_uses_thread_id_and_streams_mock_response(tmp_path):
    events = list(
        run_agent(
            "hello",
            thread_id="thread-123",
            config=Config.for_tests(working_dir=tmp_path, model="mock"),
            stream_mode=["updates"],
        )
    )

    assert events
    assert any("final_response" in str(event) for event in events)
