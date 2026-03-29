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

        # Debug settings
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.debug_llm: bool = os.getenv("DEBUG_LLM", "false").lower() == "true"
        self.langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.llm_api_key:
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
