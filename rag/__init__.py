"""RAG application layer (TZ-07).

Currently provides the LLM routing sublayer; ingestion/retrieval/generation
land in subsequent TZ-07 stages.
"""

from .llm import (
    AsyncTransport,
    CompletionOptions,
    LLMError,
    LLMProvider,
    LLMRouter,
    OllamaProvider,
    OpenAICompatProvider,
    Transport,
    build_router_from_env,
)


__all__ = [
    "AsyncTransport",
    "CompletionOptions",
    "LLMError",
    "LLMProvider",
    "LLMRouter",
    "OllamaProvider",
    "OpenAICompatProvider",
    "Transport",
    "build_router_from_env",
]
