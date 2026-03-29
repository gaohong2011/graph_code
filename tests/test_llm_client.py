"""Unit tests for llm.client module."""

from unittest.mock import patch, MagicMock

import pytest

from graph_code.llm.client import create_chat_model, get_llm
from graph_code.config import reset_config


class TestCreateChatModel:
    """Tests for create_chat_model function."""

    def test_create_chat_model_with_explicit_params(self):
        """Test creating model with explicit parameters."""
        reset_config()
        with patch.dict("os.environ", {}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                result = create_chat_model(
                    api_key="explicit-key",
                    base_url="https://explicit.com/v1",
                    model="explicit-model",
                    temperature=0.5,
                )

                mock_chat.assert_called_once_with(
                    api_key="explicit-key",
                    base_url="https://explicit.com/v1",
                    model="explicit-model",
                    temperature=0.5,
                    extra_body=None,
                )
                assert result is mock_instance

    def test_create_chat_model_with_config(self):
        """Test creating model using config values."""
        reset_config()
        env_vars = {
            "LLM_API_KEY": "config-key",
            "LLM_BASE_URL": "https://config.com/v1",
            "LLM_MODEL": "config-model",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                result = create_chat_model()

                mock_chat.assert_called_once_with(
                    api_key="config-key",
                    base_url="https://config.com/v1",
                    model="config-model",
                    temperature=0.0,
                    extra_body=None,
                )

    def test_create_chat_model_missing_api_key(self):
        """Test that ValueError is raised when API key is missing."""
        reset_config()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_chat_model()
            assert "API key is required" in str(exc_info.value)

    def test_create_chat_model_kimi_model_temperature(self):
        """Test that Kimi models use temperature=1.0."""
        reset_config()
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                create_chat_model(model="kimi-v1-8k")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["temperature"] == 1.0

    def test_create_chat_model_kimi_k25_disables_thinking(self):
        """Test that kimi-k2.5 disables thinking mode."""
        reset_config()
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                create_chat_model(model="kimi-k2.5")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["temperature"] == 1.0
                assert call_kwargs["extra_body"] == {"enable_thinking": False}

    def test_create_chat_model_k2_thinking_disables_thinking(self):
        """Test that kimi-k2-thinking-preview disables thinking mode."""
        reset_config()
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                create_chat_model(model="kimi-k2-thinking-preview")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["extra_body"] == {"enable_thinking": False}

    def test_create_chat_model_non_kimi_no_extra_body(self):
        """Test that non-Kimi models don't have extra_body."""
        reset_config()
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                create_chat_model(model="gpt-4o-mini")

                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["temperature"] == 0.0
                assert call_kwargs["extra_body"] is None


class TestGetLLM:
    """Tests for get_llm function."""

    def test_get_llm_calls_create_chat_model(self):
        """Test that get_llm creates model from config."""
        reset_config()
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=True):
            with patch("graph_code.llm.client.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                result = get_llm()

                mock_chat.assert_called_once()
                assert result is mock_instance
