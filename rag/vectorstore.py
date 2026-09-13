"""Vector store abstraction: InMemory (tests) + Qdrant (production).

Supports upsert (idempotent by ID) and search (cosine similarity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """A single vector point in the store."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result with score."""

    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    """Protocol for vector stores."""

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        """Insert or update points (idempotent by id)."""
        ...

    def search(
        self, collection: str, query_vector: list[float], top_k: int = 5,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Cosine-similarity top-k search with optional payload filter."""
        ...

    def count(self, collection: str) -> int:
        """Number of points in a collection."""
        ...

    def delete_collection(self, collection: str) -> None:
        """Drop a collection."""
        ...


class InMemoryVectorStore:
    """In-memory vector store for testing (cosine similarity)."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorPoint]] = {}

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        """Insert or update points (idempotent by id)."""
        if collection not in self._collections:
            self._collections[collection] = {}
        for point in points:
            self._collections[collection][point.id] = point

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Cosine top-k search with optional payload filter."""
        coll = self._collections.get(collection, {})
        query = np.array(query_vector)
        results: list[tuple[float, str, dict]] = []
        for pid, point in coll.items():
            if payload_filter:
                skip = any(
                    point.payload.get(k) != v
                    for k, v in payload_filter.items()
                )
                if skip:
                    continue
            vec = np.array(point.vector)
            norm = np.linalg.norm(query) * np.linalg.norm(vec)
            score = float(np.dot(query, vec) / norm) if norm > 0 else 0.0
            results.append((score, pid, point.payload))
        results.sort(key=lambda x: -x[0])
        return [
            SearchResult(id=pid, score=score, payload=payload)
            for score, pid, payload in results[:top_k]
        ]

    def count(self, collection: str) -> int:
        """Number of points in a collection."""
        return len(self._collections.get(collection, {}))

    def delete_collection(self, collection: str) -> None:
        """Drop a collection."""
        self._collections.pop(collection, None)


class QdrantVectorStore:
    """Qdrant vector store (production). Requires qdrant-client."""

    def __init__(self, url: str = "http://localhost:6333") -> None:
        from qdrant_client import QdrantClient
        self._client = QdrantClient(url=url)

    def _ensure_collection(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dim, distance=Distance.COSINE
                ),
            )

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        """Insert or update points; creates the collection if needed."""
        from qdrant_client.models import PointStruct

        if not points:
            return
        dim = len(points[0].vector)
        self._ensure_collection(collection, dim)
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=p.id, vector=p.vector, payload=p.payload
                )
                for p in points
            ],
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Cosine top-k search with optional payload filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if payload_filter:
            qfilter = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in payload_filter.items()
            ])
        res = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
        )
        return [
            SearchResult(id=str(p.id), score=p.score, payload=p.payload or {})
            for p in res.points
        ]

    def count(self, collection: str) -> int:
        """Number of points in a collection (0 if missing)."""
        try:
            info = self._client.get_collection(collection)
            return info.points_count or 0
        except Exception:
            return 0

    def delete_collection(self, collection: str) -> None:
        """Drop a collection."""
        self._client.delete_collection(collection)