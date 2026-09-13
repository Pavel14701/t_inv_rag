"""Tests for vector store and embeddings (TZ-07 волна 2)."""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import MockEmbedding
from rag.vectorstore import InMemoryVectorStore, VectorPoint


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def embeddings() -> MockEmbedding:
    return MockEmbedding(dim=64)


class TestVectorStore:
    def test_upsert_and_count(self, store: InMemoryVectorStore) -> None:
        store.upsert(
            "test", [VectorPoint(id="1", vector=[1.0, 0.0], payload={})]
        )
        assert store.count("test") == 1

    def test_upsert_idempotent(self, store: InMemoryVectorStore) -> None:
        point = VectorPoint(id="1", vector=[1.0], payload={"v": 1})
        store.upsert("test", [point])
        store.upsert("test", [point])
        assert store.count("test") == 1

    def test_search_returns_relevant(self, store: InMemoryVectorStore) -> None:
        store.upsert("test", [
            VectorPoint(
                id="rsi", vector=[1.0, 0.0, 0.0], payload={"name": "rsi"}
            ),
            VectorPoint(
                id="ema", vector=[0.0, 1.0, 0.0], payload={"name": "ema"}
            ),
        ])
        results = store.search("test", [1.0, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == "rsi"
        assert results[0].score > 0.9

    def test_search_with_payload_filter(
        self, store: InMemoryVectorStore
    ) -> None:
        store.upsert("test", [
            VectorPoint(id="a", vector=[1.0, 0.0], payload={"type": "doc"}),
            VectorPoint(id="b", vector=[1.0, 0.1], payload={"type": "case"}),
        ])
        results = store.search("test", [1.0, 0.0], top_k=2,
                               payload_filter={"type": "doc"})
        assert len(results) == 1
        assert results[0].id == "a"

    def test_delete_collection(self, store: InMemoryVectorStore) -> None:
        store.upsert("test", [VectorPoint(id="1", vector=[1.0], payload={})])
        store.delete_collection("test")
        assert store.count("test") == 0


class TestMockEmbedding:
    def test_deterministic(self, embeddings: MockEmbedding) -> None:
        v1 = embeddings.embed("RSI oversold")
        v2 = embeddings.embed("RSI oversold")
        assert v1 == v2

    def test_different_texts_different_vectors(
        self, embeddings: MockEmbedding
    ) -> None:
        v1 = embeddings.embed("RSI")
        v2 = embeddings.embed("MACD")
        assert v1 != v2

    def test_unit_vector(self, embeddings: MockEmbedding) -> None:
        vec = embeddings.embed("test")
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_dim(self, embeddings: MockEmbedding) -> None:
        assert embeddings.dim == 64
        assert len(embeddings.embed("test")) == 64