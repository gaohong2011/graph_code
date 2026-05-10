from langchain_core.messages import HumanMessage

from graph_code.agent.nodes import final_response
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_auto_memory_extraction_ignores_disabled(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="remember that I prefer terse replies")]

    update = final_response(state, config=config)

    assert update.get("memory_state") is None


def test_auto_memory_extraction_saves_explicit_remember(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.auto_memory_extraction_enabled = True
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="remember that I prefer terse replies")]

    update = final_response(state, config=config)

    assert update["memory_state"]["recent_memory_writes"]
