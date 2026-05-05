"""Tests for CLI control flow helpers."""

from types import SimpleNamespace

from rich.console import Console

from graph_code.agent.state import create_initial_state
from graph_code.main import _iter_node_outputs, _resume_until_complete, handle_graph_interrupt
from graph_code.tools.schema import ToolResultEnvelope


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


def test_resume_until_complete_handles_nested_interrupts(monkeypatch):
    """Approving one tool can lead to another interrupt before final output."""
    calls = []

    def fake_resume_graph(resume_value, thread_id, state=None):
        calls.append(resume_value)
        if len(calls) == 1:
            yield {
                "__interrupt__": [
                    FakeInterrupt(
                        {
                            "tool_name": "bash",
                            "reason": "Side-effecting tool requires approval",
                            "args": {"command": "python hello.py"},
                        }
                    )
                ]
            }
        else:
            yield {"final_response": "Hello World"}

    monkeypatch.setattr("graph_code.main.resume_graph", fake_resume_graph)
    console = Console(record=True)

    finals = _resume_until_complete(
        {"approved": True},
        "thread-1",
        create_initial_state(),
        console,
        SimpleNamespace(yes=True),
    )

    assert calls == [{"approved": True}, {"approved": True}]
    assert finals == ["Hello World"]


def test_iter_node_outputs_does_not_treat_node_name_as_final_response():
    outputs = list(_iter_node_outputs({"final_response": {"final": True}}))

    assert outputs == [{"final": True}]


def test_resume_until_complete_prints_tool_progress(monkeypatch):
    def fake_resume_graph(resume_value, thread_id, state=None):
        yield {
            "execute_tools": {
                "transition_reason": "tools_executed",
                "tool_results": [
                    ToolResultEnvelope.success(
                        "listed files",
                        tool_call_id="bash:1",
                        metadata={"tool_name": "bash"},
                    ).model_dump()
                ],
            }
        }
        yield {"call_model": {"transition_reason": "model_final_response"}}
        yield {"final_response": "Done"}

    monkeypatch.setattr("graph_code.main.resume_graph", fake_resume_graph)
    console = Console(record=True)

    _resume_until_complete(
        {"approved": True},
        "thread-1",
        create_initial_state(),
        console,
        SimpleNamespace(yes=True),
    )

    output = console.export_text()
    assert "Completed tool: bash" in output
    assert "Waiting for model response" in output


def test_handle_graph_interrupt_truncates_large_args(monkeypatch):
    monkeypatch.setattr("graph_code.main.Confirm.ask", lambda *args, **kwargs: True)
    console = Console(record=True)
    long_content = "x" * 5000

    result = handle_graph_interrupt(
        {
            "__interrupt__": [
                FakeInterrupt(
                    {
                        "tool_name": "write_file",
                        "reason": "Side-effecting tool requires approval",
                        "args": {"file_path": "tetris.html", "content": long_content},
                    }
                )
            ]
        },
        console,
    )

    output = console.export_text()
    assert result == {"approved": True}
    assert len(output) < 1500
    assert "tetris.html" in output
    assert "5000 chars" in output
    assert long_content not in output
