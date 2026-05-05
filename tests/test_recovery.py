"""Recovery behavior for model failures."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from graph_code.agent.graph import run_agent
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_transient_model_error_retries_with_budget(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    state = create_initial_state()

    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        Exception("Error code: 429 - engine_overloaded_error"),
        AIMessage(content="Recovered response"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model

    with (
        patch("graph_code.agent.nodes.get_llm", return_value=llm),
        patch("graph_code.agent.nodes.time.sleep") as sleep,
    ):
        events = list(run_agent("hello", state, "transient-retry", config=config))

    final_responses = [event["final_response"] for event in events if event.get("final_response")]
    assert final_responses[-1] == "Recovered response"
    assert bound_model.invoke.call_count == 2
    sleep.assert_called_once()
    assert state["recovery_state"]["transient_retry_budget"] == 1


def test_single_transient_retry_budget_allows_one_retry(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    state = create_initial_state()
    state["recovery_state"]["transient_retry_budget"] = 1

    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        Exception("Error code: 429 - engine_overloaded_error"),
        AIMessage(content="Recovered once"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model

    with (
        patch("graph_code.agent.nodes.get_llm", return_value=llm),
        patch("graph_code.agent.nodes.time.sleep"),
    ):
        events = list(run_agent("hello", state, "single-transient-retry", config=config))

    final_responses = [event["final_response"] for event in events if event.get("final_response")]
    assert final_responses[-1] == "Recovered once"
    assert bound_model.invoke.call_count == 2
    assert state["recovery_state"]["transient_retry_budget"] == 0
