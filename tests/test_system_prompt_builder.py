from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph_code.agent.nodes import build_prompt, call_model
from graph_code.agent.memory.paths import memory_paths_for_project
from graph_code.agent.prompt.builder import build_system_prompt
from graph_code.agent.prompt.cache import invalidate_prompt_cache
from graph_code.agent.prompt.project_instructions import load_project_instructions
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_project_instructions_load_root_to_leaf_priority(tmp_path):
    root = tmp_path / "repo"
    leaf = root / "pkg"
    leaf.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "CLAUDE.md").write_text("root instruction", encoding="utf-8")
    (leaf / "CLAUDE.md").write_text("leaf instruction", encoding="utf-8")
    config = Config.for_tests(working_dir=leaf, model="mock")

    text = load_project_instructions(config)

    assert text.index("root instruction") < text.index("leaf instruction")


def test_project_instructions_ignore_parent_outside_git_root(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("outside instruction", encoding="utf-8")
    root = tmp_path / "repo"
    leaf = root / "pkg"
    leaf.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "CLAUDE.md").write_text("repo instruction", encoding="utf-8")
    (leaf / "CLAUDE.md").write_text("leaf instruction", encoding="utf-8")
    config = Config.for_tests(working_dir=leaf, model="mock")

    text = load_project_instructions(config)

    assert "outside instruction" not in text
    assert text.index("repo instruction") < text.index("leaf instruction")


def test_project_instructions_without_git_load_only_cwd(tmp_path):
    parent = tmp_path / "parent"
    cwd = parent / "child"
    cwd.mkdir(parents=True)
    (parent / "CLAUDE.md").write_text("parent instruction", encoding="utf-8")
    (cwd / "CLAUDE.md").write_text("cwd instruction", encoding="utf-8")
    config = Config.for_tests(working_dir=cwd, model="mock")

    text = load_project_instructions(config)

    assert "parent instruction" not in text
    assert "cwd instruction" in text


def test_project_instructions_strip_crlf_frontmatter(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "CLAUDE.md").write_text("---\r\ntitle: Test\r\n---\r\nproject rule", encoding="utf-8")
    config = Config.for_tests(working_dir=tmp_path, model="mock")

    text = load_project_instructions(config)

    assert "title: Test" not in text
    assert "project rule" in text


def test_system_prompt_contains_claude_code_like_sections(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "CLAUDE.md").write_text("project rule", encoding="utf-8")
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()

    prompt = build_system_prompt(state, config)

    assert "You are Graph Code" in prompt
    assert "read code before editing" in prompt
    assert "automatic context compaction" in prompt
    assert "persistent, file-based memory system" in prompt
    assert "project rule" in prompt
    assert str(tmp_path) in prompt


def test_system_prompt_refreshes_project_instructions_without_compaction(tmp_path):
    (tmp_path / ".git").mkdir()
    instructions = tmp_path / "CLAUDE.md"
    instructions.write_text("old project rule", encoding="utf-8")
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()

    first = build_system_prompt(state, config)
    instructions.write_text("new project rule", encoding="utf-8")
    second = build_system_prompt(state, config)

    assert "old project rule" in first
    assert "new project rule" in second


def test_system_prompt_refreshes_memory_index_without_compaction(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    paths = memory_paths_for_project(config)

    first = build_system_prompt(state, config)
    paths.memory_index.write_text("- [New](new.md) - new memory text\n", encoding="utf-8")
    second = build_system_prompt(state, config)

    assert "new memory text" not in first
    assert "new memory text" in second


def test_build_prompt_stores_system_prompt(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="hello")]

    result = build_prompt(state, config=config)

    assert result["system_prompt"]
    assert result["transition_reason"] == "prompt_built"


def test_invalidate_prompt_cache_mutates_state():
    state = {"prompt_state": {"cache": {"memory": "old"}, "invalidated": False}}

    prompt_state = invalidate_prompt_cache(state)

    assert prompt_state is state["prompt_state"]
    assert state["prompt_state"]["cache"] == {}
    assert state["prompt_state"]["invalidated"] is True


def test_call_model_mock_uses_message_context_for_response(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="hello")]
    state["system_prompt"] = "CUSTOM SYSTEM PROMPT"

    result = call_model(state, config=config)

    assert result["final_response"] == "Mock response: hello"


def test_call_model_sends_built_system_prompt_to_non_mock_llm(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="real-model")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="hello")]
    state["system_prompt"] = "CUSTOM SYSTEM PROMPT"

    with patch("graph_code.agent.nodes.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        bound_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.bind_tools.return_value = bound_llm
        bound_llm.invoke.return_value = AIMessage(content="real response")

        result = call_model(state, config=config)

    captured_messages = bound_llm.invoke.call_args.args[0]
    assert isinstance(captured_messages[0], SystemMessage)
    assert captured_messages[0].content == "CUSTOM SYSTEM PROMPT"
    assert result["final_response"] == "real response"
