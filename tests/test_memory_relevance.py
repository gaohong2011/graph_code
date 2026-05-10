from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from graph_code.agent.memory.paths import memory_paths_for_project
from graph_code.agent.memory.relevance import (
    build_relevant_memory_context,
    select_relevant_memories,
)
from graph_code.agent.nodes import build_prompt
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_relevance_disabled_returns_empty(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")

    assert select_relevant_memories("testing", config) == []


def test_relevance_selects_valid_memory_files(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="real")
    config.llm_api_key = "test-key"
    config.memory_relevance_enabled = True
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True)
    (paths.memory_dir / "testing.md").write_text(
        "---\nname: testing\ndescription: Database testing policy\ntype: feedback\n---\nBody",
        encoding="utf-8",
    )

    with patch("graph_code.agent.memory.relevance.get_llm") as mock_get_llm:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content='{"selected_memories": ["testing.md"]}')
        mock_get_llm.return_value = llm

        selected = select_relevant_memories("database tests", config)

    assert [item.name for item in selected] == ["testing.md"]


def test_relevance_filters_invalid_model_choices_before_limit(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="real")
    config.llm_api_key = "test-key"
    config.memory_relevance_enabled = True
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True)
    for name in ("one.md", "two.md"):
        (paths.memory_dir / name).write_text(
            f"---\nname: {name}\ndescription: policy\ntype: feedback\n---\nBody",
            encoding="utf-8",
        )

    with patch("graph_code.agent.memory.relevance.get_llm") as mock_get_llm:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(
            content='{"selected_memories": ["missing-a.md", "missing-b.md", "one.md", "two.md"]}'
        )
        mock_get_llm.return_value = llm

        selected = select_relevant_memories("policy", config, limit=2)

    assert [item.name for item in selected] == ["one.md", "two.md"]


def test_build_relevant_memory_context_reads_selected_files(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.memory_relevance_enabled = True
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True)
    topic = paths.memory_dir / "testing.md"
    topic.write_text("---\ndescription: policy\ntype: feedback\n---\nUse real DB.", encoding="utf-8")

    context = build_relevant_memory_context([topic])

    assert "Relevant memories" in context
    assert "Use real DB" in context


def test_build_prompt_returns_surfaced_relevant_memories(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.memory_relevance_enabled = True
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True)
    topic = paths.memory_dir / "testing.md"
    topic.write_text("---\ndescription: policy\ntype: feedback\n---\nUse real DB.", encoding="utf-8")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="testing")]

    update = build_prompt(state, config=config)

    assert "Relevant memories" in update["system_prompt"]
    assert update["memory_state"]["surfaced_memories"] == [topic.as_posix()]
