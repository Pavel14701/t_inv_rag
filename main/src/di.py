"""DI providers for all system contours (TZ-08)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dishka import Provider, Scope, provide


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Root config assembled from env at startup."""

    database_url: str = ""
    rabbitmq_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    model_bundle_path: str = ""
    llm_model: str = "qwen3:8b"

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "AppConfig":
        """Build config from environment variables."""
        import os

        e = dict(os.environ if env is None else env)
        return AppConfig(
            database_url=e.get("DATABASE_URL", ""),
            rabbitmq_url=e.get("RABBITMQ_URL", ""),
            ollama_base_url=e.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            qdrant_url=e.get("QDRANT_URL", "http://localhost:6333"),
            model_bundle_path=e.get("MODEL_BUNDLE_PATH", ""),
            llm_model=e.get("LLM_MODEL", "qwen3:8b"),
        )


@dataclass(frozen=True, slots=True)
class ContourConfig:
    """Which components each contour needs."""

    name: str
    use_risk_engine: bool = False
    use_model_bundle: bool = False


CONTOURS: dict[str, ContourConfig] = {
    "backtest": ContourConfig("backtest", use_risk_engine=True),
    "inference": ContourConfig("inference", use_model_bundle=True),
    "rag": ContourConfig("rag"),
    "api": ContourConfig("api"),
}


class ConfigProvider(Provider):
    """Provides the root configuration (singleton)."""

    scope = Scope.APP

    @provide
    def app_config(self) -> AppConfig:
        """Build AppConfig from environment."""
        return AppConfig.from_env()


class RiskProvider(Provider):
    """Provides risk engine config."""

    scope = Scope.APP

    @provide
    def risk_config(self) -> Any:
        """Load RiskConfig from YAML."""
        from risk.config import load_config

        return load_config()


class LLMPort(Protocol):
    """DI key: LLM provider capability (rag contour)."""

    def complete(self, prompt: str, options: Any) -> str:
        """Synchronous completion."""
        ...


class EmbeddingsPort(Protocol):
    """DI key: embedding function capability (rag contour)."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a text into a fixed-size vector."""
        ...


class VectorStorePort(Protocol):
    """DI key: vector store capability (rag contour)."""

    def upsert(self, collection: str, points: list[Any]) -> None:
        """Insert or update points."""
        ...

    def search(
        self, collection: str, query_vector: list[float], top_k: int = 5,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Cosine top-k search."""
        ...

    def count(self, collection: str) -> int:
        """Number of points in a collection."""
        ...

    def delete_collection(self, collection: str) -> None:
        """Drop a collection."""
        ...


@dataclass(frozen=True, slots=True)
class ModelBundlePort:
    """DI key: optional AI model bundle (inference contour)."""

    bundle: Any | None = None


class LLMProviderDishka(Provider):
    """Provides the Ollama-backed LLM provider (lazy, no network on init)."""

    scope = Scope.APP

    @provide
    def llm_provider(self, config: AppConfig) -> LLMPort:
        """Build OllamaProvider from config."""
        from rag.llm import OllamaProvider

        return OllamaProvider(
            base_url=config.ollama_base_url,
            default_model=config.llm_model,
        )


class EmbeddingProvider(Provider):
    """Provides the embedding function (Ollama bge-m3)."""

    scope = Scope.APP

    @provide
    def embeddings(self, config: AppConfig) -> EmbeddingsPort:
        """Build the Ollama embedding function."""
        from rag.embeddings import OllamaEmbedding

        return OllamaEmbedding(
            base_url=config.ollama_base_url, model="bge-m3"
        )


class VectorStoreProvider(Provider):
    """Provides the vector store: Qdrant (client is lazy)."""

    scope = Scope.APP

    @provide
    def vector_store(self, config: AppConfig) -> VectorStorePort:
        """Build QdrantVectorStore from config."""
        from rag.vectorstore import QdrantVectorStore

        return QdrantVectorStore(url=config.qdrant_url)


class ModelBundleProvider(Provider):
    """Provides the AI model bundle (optional, from env path)."""

    scope = Scope.APP

    @provide
    def model_bundle(self, config: AppConfig) -> ModelBundlePort:
        """Load the bundle if MODEL_BUNDLE_PATH is set, else hold None."""
        if not config.model_bundle_path:
            return ModelBundlePort(bundle=None)
        from ai.src.bundle import load_bundle

        return ModelBundlePort(
            bundle=load_bundle(config.model_bundle_path)
        )


def build_container(contour: str) -> Any:
    """Build a dishka container for a named contour."""
    from dishka import make_container

    if contour not in CONTOURS:
        raise ValueError(
            f"Unknown contour {contour!r}; available: {sorted(CONTOURS)}"
        )
    providers: list[Provider] = [ConfigProvider()]
    if CONTOURS[contour].use_risk_engine:
        providers.append(RiskProvider())
    if CONTOURS[contour].use_model_bundle:
        providers.append(ModelBundleProvider())
    if contour in ("rag",):
        providers.extend([
            LLMProviderDishka(),
            EmbeddingProvider(),
            VectorStoreProvider(),
        ])
    return make_container(*providers)
