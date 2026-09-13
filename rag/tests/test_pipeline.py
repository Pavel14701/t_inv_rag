"""Tests for RAG retrieval and pipeline with pass@1 (TZ-07 волна 2)."""

from __future__ import annotations

import pytest

from rag.embeddings import MockEmbedding
from rag.ingestion import StrategyCase
from rag.pipeline import RAGPipeline
from rag.retrieval import CASE_COLLECTION, DOC_COLLECTION, Retriever
from rag.vectorstore import InMemoryVectorStore, VectorPoint


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def embeddings() -> MockEmbedding:
    return MockEmbedding(dim=64)


@pytest.fixture
def retriever(
    store: InMemoryVectorStore, embeddings: MockEmbedding,
) -> Retriever:
    return Retriever(store, embeddings)


class TestRetriever:
    def test_retrieve_docs(
        self, store: InMemoryVectorStore, embeddings: MockEmbedding,
        retriever: Retriever,
    ) -> None:
        store.upsert(DOC_COLLECTION, [
            VectorPoint(
                id="rsi-doc",
                vector=embeddings.embed(
                    "RSI measures momentum, oversold below 30"
                ),
                payload={"heading": "RSI Guide", "text": "RSI doc"},
            ),
        ])
        result = retriever.retrieve("RSI oversold momentum")
        assert len(result.docs) > 0
        assert result.docs[0].heading == "RSI Guide"

    def test_retrieve_cases(
        self, store: InMemoryVectorStore, embeddings: MockEmbedding,
        retriever: Retriever,
    ) -> None:
        store.upsert(CASE_COLLECTION, [
            VectorPoint(
                id="s1",
                vector=embeddings.embed("RSI oversold entry strategy"),
                payload={
                    "strategy_id": "s1",
                    "dsl_entry": "rsi.value < 30",
                    "description": "RSI entry",
                    "indicators_used": ["rsi"],
                },
            ),
        ])
        result = retriever.retrieve("RSI oversold entry")
        assert len(result.cases) > 0
        assert result.cases[0].strategy_id == "s1"

    def test_empty_store_returns_empty(self, retriever: Retriever) -> None:
        result = retriever.retrieve("anything")
        assert result.docs == []
        assert result.cases == []


def _build_pipeline(responses: list[str]) -> RAGPipeline:
    from rag.llm import LLMRouter

    store = InMemoryVectorStore()
    embeddings = MockEmbedding(dim=64)
    router = LLMRouter(default_provider="mock")
    call_count = 0

    class MockProvider:
        name = "mock"

        def complete(self, prompt, options):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        async def acomplete(self, prompt, options):
            return self.complete(prompt, options)

    router.register("mock", MockProvider())
    manifest = {
        "indicators": {
            "rsi": {"attributes": ["value"]},
            "close": {"attributes": []},
        }
    }
    return RAGPipeline(store, embeddings, router, manifest)


class TestRAGPipeline:
    def test_ingest_strategies(self) -> None:
        pipeline = _build_pipeline([])
        cases = [
            StrategyCase(
                case_id="s1", dsl_entry="rsi.value < 30", dsl_exit=None,
                indicators_used=["rsi"], description="RSI oversold",
                manifest_hash="abc",
            ),
        ]
        count = pipeline.ingest_strategies(cases)
        assert count == 1

    def test_generate_with_retrieval(self) -> None:
        pipeline = _build_pipeline(["rsi.value < 30"])
        cases = [
            StrategyCase(
                case_id="s1", dsl_entry="rsi.value < 30", dsl_exit=None,
                indicators_used=["rsi"], description="RSI oversold entry",
                manifest_hash="abc",
            ),
        ]
        pipeline.ingest_strategies(cases)
        result = pipeline.generate("RSI oversold entry strategy")
        assert result.status == "ok"
        assert result.dsl_entry == "rsi.value < 30"

    def test_pass_at_1_tracking(self) -> None:
        pipeline = _build_pipeline(["rsi.value < 30"])
        pipeline.generate("query 1")
        pipeline.generate("query 2")
        metrics = pipeline.metrics
        assert metrics.total_queries == 2
        assert metrics.pass_at_1 == 2
        assert metrics.pass_at_1_pct == 100.0

    def test_pass_at_n_with_repair(self) -> None:
        pipeline = _build_pipeline([
            "bad_ind < 1",
            "rsi.value < 30",
            "rsi.value < 30",
        ])
        pipeline.generate("query 1")
        pipeline.generate("query 2")
        metrics = pipeline.metrics
        assert metrics.total_queries == 2
        assert metrics.pass_at_1 == 1
        assert metrics.pass_at_n == 2
        assert metrics.pass_at_1_pct == 50.0

    def test_failed_queries_tracked(self) -> None:
        pipeline = _build_pipeline(["bad_ind < 1", "bad_ind > 2"])
        pipeline.generate("impossible query")
        metrics = pipeline.metrics
        assert metrics.total_queries == 1
        assert metrics.failed == 1
        assert metrics.pass_at_n == 0

    def test_query_logs_populated(self) -> None:
        pipeline = _build_pipeline(["rsi.value < 30"])
        pipeline.generate("test query")
        logs = pipeline.query_logs()
        assert len(logs) == 1
        assert logs[0].query == "test query"
        assert logs[0].status == "ok"
        assert logs[0].elapsed_ms > 0