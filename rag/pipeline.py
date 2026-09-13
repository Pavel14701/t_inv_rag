"""Full RAG pipeline: ingest, retrieve, generate, pass@1 (TZ-07 vol. 2)."""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.embeddings import EmbeddingFunction
from rag.generation import GenerationResult, generate_dsl
from rag.ingestion import (
    StrategyCase,
    ingest_docs,
)
from rag.llm import LLMRouter
from rag.retrieval import CASE_COLLECTION, DOC_COLLECTION, Retriever
from rag.vectorstore import VectorPoint, VectorStore


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """Aggregate generation quality metrics."""

    total_queries: int = 0
    pass_at_1: int = 0
    pass_at_n: int = 0
    failed: int = 0
    total_iterations: int = 0
    avg_iterations: float = 0.0

    @property
    def pass_at_1_pct(self) -> float:
        """Share of queries valid on the first attempt (percent)."""
        if not self.total_queries:
            return 0.0
        return self.pass_at_1 / self.total_queries * 100

    @property
    def pass_at_n_pct(self) -> float:
        """Share of queries valid within the repair budget (percent)."""
        if not self.total_queries:
            return 0.0
        return self.pass_at_n / self.total_queries * 100


@dataclass(slots=True)
class QueryLog:
    """Log entry for each query (for pass@1 tracking)."""

    query: str
    status: str
    iterations: int
    dsl_entry: str
    errors: list[str]
    elapsed_ms: float
    docs_used: list[str] = field(default_factory=list)


class RAGPipeline:
    """Full RAG pipeline: ingest → retrieve → generate → metrics."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingFunction,
        router: LLMRouter,
        manifest: dict[str, Any],
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._router = router
        self._manifest = manifest
        self._retriever = Retriever(store, embeddings)
        self._query_logs: list[QueryLog] = []

    # -- Ingestion ---------------------------------------------------------- #

    def ingest_docs(self, doc_paths: list[Path]) -> int:
        """Chunk and embed markdown docs into the dsl_docs collection."""
        chunks = ingest_docs(doc_paths)
        points = [
            VectorPoint(
                id=c.chunk_id,
                vector=self._embeddings.embed(c.text),
                payload={
                    "source": c.source,
                    "heading": c.heading,
                    "text": c.text,
                    "doc_type": c.doc_type,
                },
            )
            for c in chunks
        ]
        self._store.upsert(DOC_COLLECTION, points)
        return len(points)

    def ingest_strategies(self, strategies: list[StrategyCase]) -> int:
        """Embed strategy cases into the strategy_cases collection."""
        points = [
            VectorPoint(
                id=case.case_id,
                vector=self._embeddings.embed(
                    f"{case.description} {' '.join(case.indicators_used)}"
                ),
                payload={
                    "strategy_id": case.case_id,
                    "dsl_entry": case.dsl_entry,
                    "dsl_exit": case.dsl_exit,
                    "indicators_used": case.indicators_used,
                    "description": case.description,
                    "manifest_hash": case.manifest_hash,
                },
            )
            for case in strategies
        ]
        self._store.upsert(CASE_COLLECTION, points)
        return len(points)

    # -- Retrieve + Generate ------------------------------------------------ #

    def generate(self, task: str) -> GenerationResult:
        """Full RAG: retrieve context → generate DSL → validate → repair."""
        retrieval = self._retriever.retrieve(task)
        docs = [
            {"heading": d.heading, "text": d.text} for d in retrieval.docs
        ]
        examples = [
            {
                "description": c.description,
                "dsl_entry": c.dsl_entry,
            }
            for c in retrieval.cases
        ]
        start = time.perf_counter()
        result = generate_dsl(
            router=self._router,
            task=task,
            manifest=self._manifest,
            docs=docs,
            examples=examples,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._query_logs.append(QueryLog(
            query=task,
            status=result.status,
            iterations=result.iterations,
            dsl_entry=result.dsl_entry,
            errors=result.errors,
            elapsed_ms=elapsed_ms,
            docs_used=result.docs_used,
        ))
        return result

    # -- Metrics ------------------------------------------------------------ #

    @property
    def metrics(self) -> PipelineMetrics:
        """Aggregate pass@1 / pass@N metrics from query logs."""
        logs = self._query_logs
        if not logs:
            return PipelineMetrics()
        pass_at_1 = sum(
            1 for entry in logs
            if entry.status == "ok" and entry.iterations == 1
        )
        pass_at_n = sum(1 for entry in logs if entry.status == "ok")
        total_iters = sum(
            entry.iterations for entry in logs if entry.status == "ok"
        )
        avg_iters = total_iters / pass_at_n if pass_at_n else 0.0
        return PipelineMetrics(
            total_queries=len(logs),
            pass_at_1=pass_at_1,
            pass_at_n=pass_at_n,
            failed=len(logs) - pass_at_n,
            total_iterations=total_iters,
            avg_iterations=avg_iters,
        )

    def query_logs(self) -> list[QueryLog]:
        """Return all query logs for analysis."""
        return list(self._query_logs)