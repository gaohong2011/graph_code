"""Configuration management for Graph Code.

Environment variables:
    LLM_API_KEY: API key for LLM service
    LLM_BASE_URL: Base URL for LLM API (optional)
    LLM_MODEL: Model name (default: gpt-4o-mini)
    WORKING_DIR: Working directory (default: current directory)
    AUTO_CONFIRM: Skip confirmation for tools (default: false)
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Graph Code configuration."""

    def __init__(self):
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
        self.llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.working_dir: str = os.getenv("WORKING_DIR", os.getcwd())
        self.auto_confirm: bool = os.getenv("AUTO_CONFIRM", "false").lower() == "true"
        self.max_tool_iterations: int = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))
        self.permission_mode: str = os.getenv("PERMISSION_MODE", "default")
        self.output_limit: int = int(os.getenv("OUTPUT_LIMIT", "12000"))
        self.agent_data_dir: str = os.getenv("AGENT_DATA_DIR", ".agent")
        self.checkpoint_backend: str = os.getenv("CHECKPOINT_BACKEND", "memory")
        self.checkpoint_uri: Optional[str] = os.getenv("CHECKPOINT_URI")
        self.store_backend: str = os.getenv("STORE_BACKEND", "memory")
        self.store_uri: Optional[str] = os.getenv("STORE_URI")
        self.context_window_tokens: int = int(os.getenv("CONTEXT_WINDOW_TOKENS", "200000"))
        self.auto_compact_ratio: float = float(os.getenv("AUTO_COMPACT_RATIO", "0.82"))
        self.micro_compact_ratio: float = float(os.getenv("MICRO_COMPACT_RATIO", "0.68"))
        self.compact_recent_messages: int = int(os.getenv("COMPACT_RECENT_MESSAGES", "12"))
        self.micro_compact_keep_tool_results: int = int(
            os.getenv("MICRO_COMPACT_KEEP_TOOL_RESULTS", "4")
        )
        self.compact_summary_max_chars: int = int(os.getenv("COMPACT_SUMMARY_MAX_CHARS", "12000"))
        self.micro_compact_preview_chars: int = int(os.getenv("MICRO_COMPACT_PREVIEW_CHARS", "240"))
        self.micro_compact_min_tool_result_tokens: int = int(
            os.getenv("MICRO_COMPACT_MIN_TOOL_RESULT_TOKENS", "128")
        )
        self.compact_message_count_threshold: int = int(
            os.getenv("COMPACT_MESSAGE_COUNT_THRESHOLD", "40")
        )
        self.compact_use_model_summary: bool = (
            os.getenv("COMPACT_USE_MODEL_SUMMARY", "true").lower() == "true"
        )
        self.compact_warning_ratio: float = float(os.getenv("COMPACT_WARNING_RATIO", "0.65"))
        self.compact_failure_circuit_breaker: int = int(
            os.getenv("COMPACT_FAILURE_CIRCUIT_BREAKER", "3")
        )
        self.compact_summary_retry_budget: int = int(os.getenv("COMPACT_SUMMARY_RETRY_BUDGET", "1"))
        self.time_based_microcompact_turn_gap: int = int(
            os.getenv("TIME_BASED_MICROCOMPACT_TURN_GAP", "0")
        )

        # Debug settings
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.debug_llm: bool = os.getenv("DEBUG_LLM", "false").lower() == "true"
        self.langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")

    @classmethod
    def for_tests(
        cls,
        working_dir: str | Path,
        model: str = "mock",
        api_key: str = "test-key",
        permission_mode: str = "default",
    ) -> "Config":
        """Create an isolated config for tests and embedded runners."""
        config = cls.__new__(cls)
        config.llm_api_key = api_key
        config.llm_base_url = None
        config.llm_model = model
        config.working_dir = str(working_dir)
        config.auto_confirm = False
        config.max_tool_iterations = 10
        config.permission_mode = permission_mode
        config.output_limit = 12000
        config.agent_data_dir = ".agent"
        config.checkpoint_backend = "memory"
        config.checkpoint_uri = None
        config.store_backend = "memory"
        config.store_uri = None
        config.context_window_tokens = 200000
        config.auto_compact_ratio = 0.82
        config.micro_compact_ratio = 0.68
        config.compact_recent_messages = 12
        config.micro_compact_keep_tool_results = 4
        config.compact_summary_max_chars = 12000
        config.micro_compact_preview_chars = 240
        config.micro_compact_min_tool_result_tokens = 128
        config.compact_message_count_threshold = 40
        config.compact_use_model_summary = False
        config.compact_warning_ratio = 0.65
        config.compact_failure_circuit_breaker = 3
        config.compact_summary_retry_budget = 1
        config.time_based_microcompact_turn_gap = 0
        config.debug = False
        config.debug_llm = False
        config.langsmith_api_key = None
        return config

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.llm_model != "mock" and not self.llm_api_key:
            errors.append("LLM_API_KEY is required. Set it as environment variable.")
        return errors

    @property
    def working_path(self) -> Path:
        """Get working directory as Path object."""
        return Path(self.working_dir).resolve()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset config (useful for testing)."""
    global _config
    _config = None
