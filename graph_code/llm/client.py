"""LLM client for Graph Code.

Supports OpenAI-compatible APIs through base_url configuration.
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import get_config
from ..utils.debug import get_debug_callbacks


def create_chat_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create a chat model instance.

    Args:
        api_key: API key for LLM service
        base_url: Base URL for LLM API
        model: Model name
        temperature: Sampling temperature

    Returns:
        BaseChatModel instance
    """
    config = get_config()

    api_key = api_key or config.llm_api_key
    base_url = base_url or config.llm_base_url
    model = model or config.llm_model

    if not api_key:
        raise ValueError("API key is required. Set LLM_API_KEY environment variable.")

    # Handle special cases for specific models
    extra_body = None
    if model and "kimi" in model.lower():
        temperature = 1.0
        # Disable thinking mode for kimi-k2.5 to avoid reasoning_content issues
        if "k2.5" in model.lower() or "k2-thinking" in model.lower():
            extra_body = {"enable_thinking": False}

    # Get debug callbacks if debugging is enabled
    callbacks = get_debug_callbacks()

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        extra_body=extra_body,
        callbacks=callbacks,
    )


def get_llm() -> BaseChatModel:
    """Get the default LLM instance from config."""
    return create_chat_model()
