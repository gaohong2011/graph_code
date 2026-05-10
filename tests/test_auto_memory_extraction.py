import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from graph_code.agent.memory.paths import memory_paths_for_project
from graph_code.agent.nodes import final_response
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_auto_memory_extraction_ignores_disabled(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="remember that I prefer terse replies")]

    update = final_response(state, config=config)

    assert update.get("memory_state") is None


def test_auto_memory_extraction_ignores_memory_disabled(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.auto_memory_extraction_enabled = True
    config.memory_disabled = True
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


def test_auto_memory_extraction_uses_distinct_keys_for_distinct_memories(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.auto_memory_extraction_enabled = True
    paths = memory_paths_for_project(config)

    first = create_initial_state()
    first["messages"] = [HumanMessage(content="remember that I prefer terse replies")]
    first_update = final_response(first, config=config)

    second = create_initial_state()
    second["messages"] = [HumanMessage(content="remember that I prefer python tests")]
    second_update = final_response(second, config=config)

    first_path = json.loads(first_update["memory_state"]["recent_memory_writes"][0])["path"]
    second_path = json.loads(second_update["memory_state"]["recent_memory_writes"][0])["path"]
    assert first_path != second_path
    assert len(list(paths.memory_dir.glob("feedback_explicit_user_memory_*.md"))) == 2
    assert "I prefer terse replies" in Path(first_path).read_text(encoding="utf-8")


def test_auto_memory_extraction_requires_start_marker_and_stores_content_only(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.auto_memory_extraction_enabled = True
    ignored = create_initial_state()
    ignored["messages"] = [HumanMessage(content="the docs say remember that foo")]

    ignored_update = final_response(ignored, config=config)

    assert ignored_update.get("memory_state") is None

    state = create_initial_state()
    state["messages"] = [HumanMessage(content="  remember that I prefer terse replies")]

    update = final_response(state, config=config)

    write_path = json.loads(update["memory_state"]["recent_memory_writes"][0])["path"]
    content = Path(write_path).read_text(encoding="utf-8")
    assert "I prefer terse replies" in content
    assert "remember that I prefer terse replies" not in content


def test_final_response_existing_response_keeps_previous_update_when_extraction_disabled(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["final_response"] = "already done"
    state["messages"] = [HumanMessage(content="remember that I prefer terse replies")]

    update = final_response(state, config=config)

    assert update == {"final": True}


def test_final_response_existing_response_merges_memory_without_redundant_response(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.auto_memory_extraction_enabled = True
    state = create_initial_state()
    state["final_response"] = "already done"
    state["messages"] = [HumanMessage(content="remember that I prefer terse replies")]

    update = final_response(state, config=config)

    assert update["final"] is True
    assert "final_response" not in update
    assert update["memory_state"]["recent_memory_writes"]
