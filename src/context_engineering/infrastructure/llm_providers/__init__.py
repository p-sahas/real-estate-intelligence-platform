"""
LLM providers - factory functions for models.

Supports multiple providers via OpenRouter unified API:
- OpenAI (GPT-4, GPT-4o, o1, o3)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini 2.0)
- Meta (Llama 3)
- DeepSeek, Mistral, and many more

Usage:
    from context_engineering.infrastructure.llm_providers import get_chat_llm, get_default_embeddings
    
    # Use default model from config
    llm = get_chat_llm()
    
    # Use specific model via OpenRouter
    llm = get_chat_llm(model="anthropic/claude-3-5-sonnet")
    
    # Use reasoning model
    llm = get_chat_llm(tier="reason")
"""

from .llm_services import (
    get_chat_llm,
    get_reasoning_llm,
    get_strong_llm,
    list_available_models,
)
from .embeddings import (
    get_default_embeddings,
    get_small_embeddings,
)

__all__ = [
    # Chat LLMs
    "get_chat_llm",
    "get_reasoning_llm",
    "get_strong_llm",
    "list_available_models",
    # Embeddings
    "get_default_embeddings",
    "get_small_embeddings",
]
