"""
Infrastructure layer - external integrations.

Contains:
- llm_providers: LLM and embedding services
- db: Database and storage
- api: API endpoints
- monitoring: Logging and metrics
"""

from .llm_providers import get_chat_llm, get_default_embeddings

__all__ = [
    "get_chat_llm",
    "get_default_embeddings",
]
