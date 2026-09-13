"""RAG retrieval: search docs and cases in the vector store (TZ-07)."""

from __future__ import annotations

from dataclasses import dataclass

from rag.embeddings import EmbeddingFunction
from rag.vectorstore import VectorStore


DOC_COLLECTION = "dsl_docs"
CASE_COLLECTION = "strategy_cases"


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    """A retrieved documentation chunk."""

    heading: str
    text: str
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedCase:
    """A retrieved strategy case."""

    strategy_id: str
    dsl_entry: str
    dsl_exit: str | None
    description: str
    indicators_used: list[str]
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Combined retrieval result for the generation prompt."""

    docs: list[RetrievedDoc]
    cases: list[RetrievedCase]


class Retriever:
    """Retrieves relevant docs and strategy cases for a query."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingFunction,
        docs_top_k: int = 5,
        cases_top_k: int = 3,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._docs_top_k = docs_top_k
        self._cases_top_k = cases_top_k

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant docs and strategy cases for a query."""
        query_vec = self._embeddings.embed(query)
        docs = self._retrieve_docs(query_vec)
        cases = self._retrieve_cases(query_vec)
        return RetrievalResult(docs=docs, cases=cases)

    def _retrieve_docs(self, query_vec: list[float]) -> list[RetrievedDoc]:
        results = self._store.search(
            DOC_COLLECTION, query_vec, top_k=self._docs_top_k
        )
        return [
            RetrievedDoc(
                heading=r.payload.get("heading", ""),
                text=r.payload.get("text", ""),
                source=r.payload.get("source", ""),
                score=r.score,
            )
            for r in results
        ]

    def _retrieve_cases(self, query_vec: list[float]) -> list[RetrievedCase]:
        results = self._store.search(
            CASE_COLLECTION, query_vec, top_k=self._cases_top_k
        )
        return [
            RetrievedCase(
                strategy_id=r.payload.get("strategy_id", ""),
                dsl_entry=r.payload.get("dsl_entry", ""),
                dsl_exit=r.payload.get("dsl_exit"),
                description=r.payload.get("description", ""),
                indicators_used=r.payload.get("indicators_used", []),
                score=r.score,
            )
            for r in results
        ]