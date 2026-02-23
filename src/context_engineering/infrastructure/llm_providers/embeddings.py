"""
Embedding model provider with OpenRouter support.

Provides unified access to embedding models across providers.
Currently supports OpenAI embeddings (text-embedding-3-large/small).

Note: OpenRouter also supports embeddings via the same unified API.
"""

from typing import Optional, Any
import os
from langchain_openai import OpenAIEmbeddings

from context_engineering.config import (
    PROVIDER,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_SHOW_PROGRESS,
    get_api_key,
    get_embedding_model,
)


def get_default_embeddings(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tier: str = "default",
    batch_size: Optional[int] = None,
    show_progress: Optional[bool] = None,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """
    Factory function to create an embedding model instance.
    
    Supports multiple providers via OpenRouter unified API,
    or direct OpenAI access if configured.
    
    Args:
        model: Model name (e.g., "text-embedding-3-large"). If None, uses config.
        provider: Override provider ("openrouter", "openai")
        tier: Embedding tier ("default", "small")
        batch_size: Number of texts to embed in parallel
        show_progress: Display progress bar for large batch operations
        **kwargs: Additional provider-specific parameters
    
    Returns:
        OpenAIEmbeddings: An embedding model instance ready for vectorization
    
    Examples:
        # Use default from config
        >>> embedder = get_default_embeddings()
        
        # Use smaller/cheaper model
        >>> embedder = get_default_embeddings(tier="small")
        
        # Use specific model
        >>> embedder = get_default_embeddings(model="text-embedding-3-small")
    
    Dimensions:
        - text-embedding-3-large: 3072 dimensions
        - text-embedding-3-small: 1536 dimensions
    """
    # Determine provider
    use_provider = provider or PROVIDER
    
    # Determine model
    if model:
        use_model = model
    else:
        use_model = get_embedding_model(provider=use_provider, tier=tier)
    
    # Strip provider prefix if present (embeddings use bare model name)
    if "/" in use_model:
        use_model = use_model.split("/")[-1]
    
    # Get API key
    api_key = get_api_key(use_provider)
    if not api_key:
        # Fallback to OPENAI_API_KEY for backward compatibility
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Set defaults
    use_batch_size = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE
    use_show_progress = show_progress if show_progress is not None else EMBEDDING_SHOW_PROGRESS
    
    # Configure based on provider
    if use_provider == "openrouter":
        # OpenRouter embeddings (uses OpenAI-compatible API)
        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            show_progress_bar=use_show_progress,
            **kwargs
        )
    else:
        # Direct OpenAI access
        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            show_progress_bar=use_show_progress,
            **kwargs
        )


def get_small_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """Get smaller/cheaper embedding model (1536 dimensions)."""
    return get_default_embeddings(tier="small", **kwargs)
