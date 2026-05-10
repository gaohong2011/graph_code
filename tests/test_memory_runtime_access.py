import json
import subprocess
import sys
from pathlib import Path

from graph_code.agent.memory.paths import memory_paths_for_project
from graph_code.config import Config
from graph_code.tools.runtime import ToolExecutionRuntime


def test_runtime_imports_in_fresh_process():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from graph_code.tools.runtime import ToolExecutionRuntime; print(ToolExecutionRuntime.__name__)",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ToolExecutionRuntime" in result.stdout


def test_runtime_can_access_configured_memory_dir_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = tmp_path / "memory-root"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("memory index", encoding="utf-8")
    config = Config.for_tests(working_dir=workspace, model="mock")
    config.memory_dir = str(memory)

    runtime = ToolExecutionRuntime(workspace, config=config)

    result = runtime.execute(
        [{"id": "read-memory", "name": "read_file", "args": {"file_path": str(memory / "MEMORY.md")}}],
        skip_permissions=True,
    )[0]

    assert result.ok is True
    assert "memory index" in result.content


def test_runtime_rejects_non_memory_home_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = tmp_path / "memory-root"
    memory.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    config = Config.for_tests(working_dir=workspace, model="mock")
    config.memory_dir = str(memory)

    runtime = ToolExecutionRuntime(workspace, config=config)

    result = runtime.execute(
        [{"id": "read-outside", "name": "read_file", "args": {"file_path": str(outside)}}],
        skip_permissions=True,
    )[0]

    assert result.ok is False
    assert "outside working directory" in result.content


def test_runtime_can_search_configured_memory_dir_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = tmp_path / "memory-root"
    memory.mkdir()
    (memory / "policy.md").write_text("Use real DB.", encoding="utf-8")
    config = Config.for_tests(working_dir=workspace, model="mock")
    config.memory_dir = str(memory)

    runtime = ToolExecutionRuntime(workspace, config=config)

    result = runtime.execute(
        [{"id": "search-memory", "name": "search_files", "args": {"path": str(memory), "pattern": "real DB"}}],
        skip_permissions=True,
    )[0]

    assert result.ok is True
    assert "Use real DB." in result.content
    assert "ValueError" not in result.content


def test_legacy_save_memory_writes_topic_and_index(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.for_tests(working_dir=workspace, model="mock")
    config.graph_code_home = str(tmp_path / "home")
    runtime = ToolExecutionRuntime(workspace, config=config)

    result = runtime.execute(
        [
            {
                "id": "save-memory",
                "name": "save_memory",
                "args": {"namespace": "feedback", "key": "testing policy", "value": "Use real DB."},
            }
        ],
        skip_permissions=True,
    )[0]

    paths = memory_paths_for_project(config)
    assert result.ok is True
    assert (paths.memory_dir / "feedback_testing_policy.md").exists()
    assert "feedback_testing_policy.md" in paths.memory_index.read_text(encoding="utf-8")
    payload = json.loads(result.content)
    assert payload["path"].endswith("feedback_testing_policy.md")
