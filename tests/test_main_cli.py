"""Tests for CLI control flow helpers."""

from types import SimpleNamespace

from rich.console import Console

from graph_code.agent.state import create_initial_state
from graph_code.main import _iter_node_outputs, _resume_until_complete


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
