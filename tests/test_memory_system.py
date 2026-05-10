from pathlib import Path

from graph_code.agent.memory.paths import memory_paths_for_project, validate_memory_root
from graph_code.agent.memory.prompt import build_memory_prompt, load_memory_index_context
from graph_code.agent.memory.scan import scan_memory_headers
from graph_code.config import Config


def test_memory_path_uses_graph_code_home_and_project_slug(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    config = Config.for_tests(working_dir=project, model="mock")
    config.graph_code_home = str(tmp_path / "home")

    paths = memory_paths_for_project(config)

    assert paths.memory_dir.parent.name != ""
    assert paths.memory_dir.name == "memory"
    assert str(paths.memory_dir).startswith(str(tmp_path / "home" / "projects"))
    assert paths.memory_index.name == "MEMORY.md"


def test_memory_override_must_be_safe(tmp_path):
    safe = tmp_path / "safe-memory"
    assert validate_memory_root(str(safe)) == safe.resolve()

    for unsafe in ["", "/", ".", "relative/path", str(Path.home())]:
        assert validate_memory_root(unsafe) is None


def test_memory_override_rejects_null_bytes_and_existing_files(tmp_path):
    existing_file = tmp_path / "memory-file"
    existing_file.write_text("not a directory", encoding="utf-8")

    assert validate_memory_root(f"{tmp_path}\0memory") is None
    assert validate_memory_root(str(existing_file)) is None


def test_memory_paths_fall_back_when_override_points_to_file(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    memory_file = tmp_path / "memory-file"
    memory_file.write_text("not a directory", encoding="utf-8")
    config = Config.for_tests(working_dir=project, model="mock")
    config.graph_code_home = str(tmp_path / "home")
    config.memory_dir = str(memory_file)

    paths = memory_paths_for_project(config)

    assert paths.memory_dir != memory_file.resolve()
    assert str(paths.memory_dir).startswith(str(tmp_path / "home" / "projects"))


def test_scan_memory_headers_reads_frontmatter(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("- [Testing](testing.md) - policy\n", encoding="utf-8")
    (memory_dir / "testing.md").write_text(
        "---\n"
        "name: testing policy\n"
        "description: Use real database in integration tests\n"
        "type: feedback\n"
        "updated_at: 2026-05-06\n"
        "---\n"
        "Body\n",
        encoding="utf-8",
    )

    headers = scan_memory_headers(memory_dir)

    assert len(headers) == 1
    assert headers[0].filename == "testing.md"
    assert headers[0].description == "Use real database in integration tests"
    assert headers[0].memory_type == "feedback"


def test_scan_memory_headers_normalizes_memory_type(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "testing.md").write_text(
        "---\n"
        "description: Use real database in integration tests\n"
        "type: Feedback \n"
        "---\n"
        "Body\n",
        encoding="utf-8",
    )

    headers = scan_memory_headers(memory_dir)

    assert len(headers) == 1
    assert headers[0].memory_type == "feedback"


def test_memory_prompt_can_be_disabled(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.memory_disabled = True

    assert build_memory_prompt(config) is None


def test_memory_prompt_creates_missing_memory_dir_and_index(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.graph_code_home = str(tmp_path / "home")
    paths = memory_paths_for_project(config)

    assert not paths.memory_dir.exists()
    assert not paths.memory_index.exists()

    prompt = build_memory_prompt(config)

    assert prompt is not None
    assert paths.memory_dir.is_dir()
    assert paths.memory_index.is_file()


def test_memory_prompt_includes_taxonomy_and_index(tmp_path):
    config = Config.for_tests(working_dir=tmp_path, model="mock")
    config.graph_code_home = str(tmp_path / "home")
    paths = memory_paths_for_project(config)
    paths.memory_dir.mkdir(parents=True)
    paths.memory_index.write_text("- [Testing](testing.md) - policy\n", encoding="utf-8")

    prompt = build_memory_prompt(config)
    index_context = load_memory_index_context(config)

    assert prompt is not None
    assert "persistent, file-based memory system" in prompt
    assert "type: feedback" in prompt
    assert "What not to save" in prompt
    assert "Testing" in index_context
