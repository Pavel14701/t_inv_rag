"""Tests for DI container assembly per contour (TZ-08 acceptance)."""

from __future__ import annotations

import pytest

from main.src.di import (
    CONTOURS,
    AppConfig,
    EmbeddingsPort,
    LLMPort,
    ModelBundlePort,
    VectorStorePort,
    build_container,
)


class TestContours:
    def test_backtest_container_builds(self) -> None:
        """Backtest contour: config + risk + ta (no torch)."""
        container = build_container("backtest")
        cfg = container.get(AppConfig)
        assert isinstance(cfg, AppConfig)

    def test_inference_container_builds(self) -> None:
        container = build_container("inference")
        cfg = container.get(AppConfig)
        assert isinstance(cfg, AppConfig)

    def test_rag_container_builds(self) -> None:
        container = build_container("rag")
        cfg = container.get(AppConfig)
        assert isinstance(cfg, AppConfig)

    def test_api_container_builds(self) -> None:
        container = build_container("api")
        cfg = container.get(AppConfig)
        assert isinstance(cfg, AppConfig)

    def test_unknown_contour_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown contour"):
            build_container("nonexistent")

    def test_all_contours_defined(self) -> None:
        assert set(CONTOURS) == {"backtest", "inference", "rag", "api"}


class TestAppConfig:
    def test_from_env_defaults(self) -> None:
        cfg = AppConfig.from_env(env={})
        assert cfg.database_url == ""
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.qdrant_url == "http://localhost:6333"

    def test_from_env_overrides(self) -> None:
        cfg = AppConfig.from_env(
            env={
                "DATABASE_URL": "postgres://test",
                "QDRANT_URL": "http://q:6333",
                "MODEL_BUNDLE_PATH": "/tmp/bundle",
                "LLM_MODEL": "llama3",
            }
        )
        assert cfg.database_url == "postgres://test"
        assert cfg.qdrant_url == "http://q:6333"
        assert cfg.model_bundle_path == "/tmp/bundle"
        assert cfg.llm_model == "llama3"

    def test_frozen(self) -> None:
        cfg = AppConfig()
        with pytest.raises(AttributeError):
            cfg.database_url = "x"  # type: ignore[misc]


class TestWave2Providers:
    """TZ-08 волна 2: rag/LLM/bundle providers."""

    def test_rag_provides_vector_store(self) -> None:
        """Rag contour wires a Qdrant vector store (lazy client)."""
        from rag.vectorstore import QdrantVectorStore

        container = build_container("rag")
        store = container.get(VectorStorePort)
        assert isinstance(store, QdrantVectorStore)

    def test_rag_provides_embeddings(self) -> None:
        from rag.embeddings import OllamaEmbedding

        container = build_container("rag")
        emb = container.get(EmbeddingsPort)
        assert isinstance(emb, OllamaEmbedding)

    def test_rag_provides_llm(self) -> None:
        from rag.llm import OllamaProvider

        container = build_container("rag")
        llm = container.get(LLMPort)
        assert isinstance(llm, OllamaProvider)

    def test_inference_model_bundle_none_without_path(self) -> None:
        """No MODEL_BUNDLE_PATH -> bundle is None, no torch import."""
        container = build_container("inference")
        port = container.get(ModelBundlePort)
        assert port.bundle is None

    def test_api_contour_has_no_rag_providers(self) -> None:
        """API contour stays free of GPU/rag dependencies (TZ-09)."""
        container = build_container("api")
        with pytest.raises(Exception):  # noqa: B017
            container.get(VectorStorePort)
