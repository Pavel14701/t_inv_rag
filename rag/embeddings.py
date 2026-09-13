"""Embedding functions: Ollama (production) + hash-based mock (tests)."""

from __future__ import annotations

import hashlib

from typing import Protocol

import numpy as np


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def embed(self, text: str) -> list[float]:
        """Embed a text string into a fixed-size vector."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class MockEmbedding:
    """Deterministic hash-based embedding for testing (no network)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed a text deterministically into a unit vector."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Generate dim floats from hash bytes (deterministic)
        vec = [
            float(digest[i % len(digest)]) / 255.0 - 0.5
            for i in range(self._dim)
        ]
        # Normalize to unit vector
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OllamaEmbedding:
    """Embedding via Ollama /api/embeddings endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
    ) -> None:
        import niquests
        self._session = niquests.Session()
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._model = model

    @property
    def dim(self) -> int:
        """Embedding dimension (bge-m3)."""
        return 1024

    def embed(self, text: str) -> list[float]:
        """Embed a text via the Ollama embeddings API."""
        resp = self._session.post(
            self._url,
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]