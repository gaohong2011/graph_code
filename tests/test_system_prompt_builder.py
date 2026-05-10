from langchain_core.messages import HumanMessage

from graph_code.agent.nodes import build_prompt, call_model
from graph_code.agent.prompt.builder import build_system_prompt
from graph_code.agent.prompt.project_instructions import load_project_instructions
from graph_code.agent.state import create_initial_state
from graph_code.config import Config


def test_project_instructions_load_root_to_leaf_priority(tmp_path):
    root = tmp_path / "repo"
    leaf = root / "pkg"
    leaf.mkdir(parents=True)
    (root / "CLAUDE.md").write_text("root instruction", encoding="utf-8")
    (leaf / "CLAUDE.md").write_text("leaf instruction", encoding="utf-8")
    config = Config.for_tests(working_dir=leaf, model="mock")

    text = load_project_instructions(config)

    assert text.index("root instruction") < text.index("leaf instruction")


def test_system_prompt_contains_claude_code_like_sections(tmp_path):
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


def test_build_prompt_stores_system_prompt(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="hello")]

    result = build_prompt(state, config=config)

    assert result["system_prompt"]
    assert result["transition_reason"] == "prompt_built"


def test_call_model_uses_built_system_prompt(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    state = create_initial_state()
    state["messages"] = [HumanMessage(content="hello")]
    state["system_prompt"] = "CUSTOM SYSTEM PROMPT"

    result = call_model(state, config=config)

    assert result["final_response"] == "Mock response: hello"
