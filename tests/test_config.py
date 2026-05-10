"""Unit tests for config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_code.config import Config, get_config, reset_config


def test_memory_and_session_config_defaults(tmp_path):
    with patch.dict(os.environ, {"WORKING_DIR": str(tmp_path)}, clear=True):
        config = Config()

    assert config.graph_code_home.endswith(".graph-code")
    assert config.memory_disabled is False
    assert config.memory_dir is None
    assert config.memory_relevance_enabled is False
    assert config.session_memory_enabled is False
    assert config.auto_memory_extraction_enabled is False
    assert config.session_memory_init_tokens == 10000
    assert config.session_memory_update_tokens == 5000
    assert config.session_memory_tool_calls == 3


def test_memory_and_session_config_from_env(tmp_path):
    env_vars = {
        "WORKING_DIR": str(tmp_path),
        "GRAPH_CODE_HOME": str(tmp_path / "home"),
        "GRAPH_CODE_MEMORY_DIR": str(tmp_path / "mem"),
        "GRAPH_CODE_DISABLE_MEMORY": "true",
        "ENABLE_MEMORY_RELEVANCE": "true",
        "ENABLE_SESSION_MEMORY": "true",
        "ENABLE_AUTO_MEMORY_EXTRACTION": "true",
        "SESSION_MEMORY_INIT_TOKENS": "123",
        "SESSION_MEMORY_UPDATE_TOKENS": "45",
        "SESSION_MEMORY_TOOL_CALLS": "6",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = Config()

    assert config.graph_code_home == str(tmp_path / "home")
    assert config.memory_dir == str(tmp_path / "mem")
    assert config.memory_disabled is True
    assert config.memory_relevance_enabled is True
    assert config.session_memory_enabled is True
    assert config.auto_memory_extraction_enabled is True
    assert config.session_memory_init_tokens == 123
    assert config.session_memory_update_tokens == 45
    assert config.session_memory_tool_calls == 6


class TestConfig:
    """Tests for Config class."""

    def test_config_defaults(self):
        """Test that Config has correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.llm_api_key is None
            assert config.llm_base_url is None
            assert config.llm_model == "gpt-4o-mini"
            assert config.auto_confirm is False
            assert config.max_tool_iterations == 10

    def test_config_from_env(self):
        """Test that Config reads from environment variables."""
        env_vars = {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://api.test.com/v1",
            "LLM_MODEL": "test-model",
            "WORKING_DIR": "/test/workdir",
            "AUTO_CONFIRM": "true",
            "MAX_TOOL_ITERATIONS": "20",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.llm_api_key == "test-key"
            assert config.llm_base_url == "https://api.test.com/v1"
            assert config.llm_model == "test-model"
            assert config.working_dir == "/test/workdir"
            assert config.auto_confirm is True
            assert config.max_tool_iterations == 20

    def test_config_auto_confirm_variations(self):
        """Test various AUTO_CONFIRM values."""
        for value, expected in [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
            ("yes", False),
            ("", False),
        ]:
            with patch.dict(os.environ, {"AUTO_CONFIRM": value}, clear=True):
                config = Config()
                assert config.auto_confirm == expected, f"Failed for value: {value}"

    def test_working_path_property(self, tmp_path):
        """Test working_path property returns resolved Path."""
        with patch.dict(os.environ, {"WORKING_DIR": str(tmp_path)}, clear=True):
            config = Config()
            assert isinstance(config.working_path, Path)
            assert config.working_path == tmp_path.resolve()

    def test_validate_with_missing_api_key(self):
        """Test validation fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            errors = config.validate()
            assert len(errors) == 1
            assert "LLM_API_KEY is required" in errors[0]

    def test_validate_with_api_key(self):
        """Test validation passes when API key is present."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = Config()
            errors = config.validate()
            assert len(errors) == 0


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_get_config_creates_new_if_none(self):
        """Test that get_config creates new config if none exists."""
        reset_config()
        with patch.dict(os.environ, {"LLM_API_KEY": "new-key"}, clear=True):
            config = get_config()
            assert config.llm_api_key == "new-key"


class TestResetConfig:
    """Tests for reset_config function."""

    def test_reset_config_clears_instance(self):
        """Test that reset_config clears the global config."""
        with patch.dict(os.environ, {"LLM_API_KEY": "first-key"}, clear=True):
            reset_config()
            config1 = get_config()
            assert config1.llm_api_key == "first-key"

            # Reset and create with different env
            reset_config()

        with patch.dict(os.environ, {"LLM_API_KEY": "second-key"}, clear=True):
            config2 = get_config()
            assert config2.llm_api_key == "second-key"
            # Should be different instances
            assert config1 is not config2
