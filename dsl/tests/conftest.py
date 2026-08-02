"""Pytest fixtures for DSL test suite.

This module provides reusable fixtures for testing the DSL interpreter,
including mock providers, contexts, and interpreters for both synchronous
and asynchronous execution modes.
"""

import pytest
from typing import Any

from dsl.context import Context
from dsl.interpreter import Interpreter
from dsl.providers.base import IndicatorProvider, AsyncIndicatorProvider
from dsl.exceptions import ProviderError
from dsl.providers.manifest import IndicatorSchema, Manifest, ParameterSchema


DEFAULT_MANIFEST = {
    'indicators': {
        'close': {'attributes': []},
        'volume': {'attributes': []},
        'low': {'attributes': []},
        'high': {'attributes': []},
        'rsi': {
            'attributes': ['value', 'signal'],
            'parameters': {'period': {'type': 'float', 'default': 14.0}}
        },
        'macd': {
            'attributes': ['line', 'signal', 'histogram'],
            'parameters': {
                'fast': {'type': 'float', 'default': 12.0},
                'slow': {'type': 'float', 'default': 26.0}
            }
        }
    }
}


class SyncMockProvider(IndicatorProvider):
    """Synchronous mock provider that returns predefined values.

    This provider is used in tests to simulate indicator resolution
    without external dependencies. It can return values for specific
    indicator/params/attributes/offset combinations and also supports
    history resolution.

    Attributes:
        values: Dictionary mapping (indicator, params_key, attrs_key, offset)
            to float values.
        history: Dictionary mapping (indicator, params_key, attrs_key) to
            list of historical values (oldest to newest).

    """

    def __init__(
        self,
        values: dict[tuple, float] | None = None,
        history: dict[tuple, list[float]] | None = None,
        manifest: dict[str, Any] | None = None
    ) -> None:
        """Initialize the mock provider.

        Args:
            values: Predefined values for indicator resolution.
            history: Predefined historical data.
            manifest: Manifest describing available indicators
                (defaults to DEFAULT_MANIFEST).

        """
        self.values = values or {}
        self.history = history or {}
        self._manifest = manifest or DEFAULT_MANIFEST

    def get_manifest(self) -> dict[str, Any]:
        """Return the manifest of available indicators."""
        return self._manifest

    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int
    ) -> float:
        """Resolve a value from the predefined values dictionary.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attributes.
            offset: Bar offset.

        Returns:
            The predefined value.

        Raises:
            ProviderError: If the key is not found.

        """
        key = (
            indicator,
            tuple(sorted(params.items())),
            tuple(attributes),
            offset
        )
        if key in self.values:
            return self.values[key]
        raise ProviderError(f'No value for {key}')

    def resolve_history(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        n: int
    ) -> list[float]:
        """Resolve historical values from the predefined history.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attributes.
            n: Number of bars.

        Returns:
            List of historical values (last n bars).

        Raises:
            ProviderError: If the key is not found.

        """
        key = (indicator, tuple(sorted(params.items())), tuple(attributes))
        if key in self.history:
            hist = self.history[key]
            return hist[-n:] if len(hist) >= n else hist
        raise ProviderError(f'No history for {key}')


class AsyncMockProvider(AsyncIndicatorProvider):
    """Asynchronous mock provider that returns predefined values.

    This is the async counterpart of SyncMockProvider. It provides
    asynchronous methods for indicator resolution and history retrieval.
    """

    def __init__(
        self,
        values: dict[tuple, float] | None = None,
        history: dict[tuple, list[float]] | None = None,
        manifest: dict[str, Any] | None = None
    ) -> None:
        self.values = values or {}
        self.history = history or {}
        self._manifest = manifest or DEFAULT_MANIFEST

    # Synchronous method required by Context during initialization
    def get_manifest(self) -> dict[str, Any]:  # noqa: D102
        return self._manifest

    async def get_manifest_async(self) -> dict[str, Any]:  # noqa: D102
        return self._manifest

    async def resolve_async(  # noqa: D102
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int
    ) -> float:
        key = (
            indicator,
            tuple(sorted(params.items())),
            tuple(attributes),
            offset
        )
        if key in self.values:
            return self.values[key]
        raise ProviderError(f'No value for {key}')

    async def resolve_history_async(  # noqa: D102
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        n: int
    ) -> list[float]:
        key = (indicator, tuple(sorted(params.items())), tuple(attributes))
        if key in self.history:
            hist = self.history[key]
            return hist[-n:] if len(hist) >= n else hist
        raise ProviderError(f'No history for {key}')


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sync_mock_provider() -> SyncMockProvider:
    """Return a synchronous mock provider with empty data."""
    return SyncMockProvider()


@pytest.fixture
def async_mock_provider() -> AsyncMockProvider:
    """Return an asynchronous mock provider with empty data."""
    return AsyncMockProvider()


@pytest.fixture
def context_empty() -> Context:
    """Return a Context with no providers."""
    return Context([])


@pytest.fixture
def context_with_sync_provider(
    sync_mock_provider: SyncMockProvider
) -> Context:
    """Return a Context with a single synchronous mock provider."""
    return Context([sync_mock_provider])


@pytest.fixture
def context_with_async_provider(
    async_mock_provider: AsyncMockProvider
) -> Context:
    """Return a Context with a single asynchronous mock provider."""
    return Context([async_mock_provider])


@pytest.fixture
def interpreter_sync(context_with_sync_provider: Context) -> Interpreter:
    """Return a synchronous interpreter with a context
    containing a sync mock provider.
    """
    return Interpreter(context_with_sync_provider)


@pytest.fixture
def interpreter_async(context_with_async_provider: Context) -> Interpreter:
    """Return an asynchronous interpreter with a
    context containing an async mock provider.
    """
    return Interpreter(context_with_async_provider)


@pytest.fixture
def context_with_history() -> Context:
    """Return a Context with history data for the 'close' indicator.

    Provides values for offset 0 and offset 1, allowing testing of
    historical access without a full manifest.
    """
    values = {
        ('close', (), (), 0): 100.0,
        ('close', (), (), 1): 95.0,
    }
    provider = SyncMockProvider(values, manifest=DEFAULT_MANIFEST)
    return Context([provider])


@pytest.fixture
def sample_values() -> dict[tuple, float]:
    """Return a typical set of indicator values for testing.

    Includes 'close', 'volume', 'rsi', 'macd', 'low', and 'high'
    indicators with various parameters and offsets.
    """
    return {
        ('close', (), (), 0): 100.0,
        ('volume', (), (), 0): 1000.0,
        ('rsi', (('period', 14),), ('value',), 0): 80.0,
        ('macd', (('fast', 12), ('slow', 26)), ('line',), 0): 1.5,
        ('macd', (('fast', 12), ('slow', 26)), ('signal',), 0): 0.5,
        ('close', (), (), 1): 95.0,
        ('low', (), (), 0): 50.0,
        ('high', (), (), 0): 110.0,
    }


@pytest.fixture
def sample_history() -> dict[tuple, list[float]]:
    """Return a typical set of historical data for testing.

    Provides historical values for 'close' and 'volume' indicators.
    """
    return {
        ('close', (), ()): [10, 20, 30, 40],
        ('volume', (), ()): [100, 90, 80, 70],
    }


@pytest.fixture
def context_with_sample_data(sample_values, sample_history) -> Context:
    """Return a Context with a sync mock provider pre-filled with sample data.

    The provider contains both current values and history for testing
    indicators, parameters, and rising/falling functions.
    """
    provider = SyncMockProvider(
        sample_values,
        sample_history,
        manifest=DEFAULT_MANIFEST
    )
    return Context([provider])


@pytest.fixture
def async_context_with_sample_data(sample_values, sample_history) -> Context:
    """Return a Context with an async mock provider pre-filled
    with sample data.

    This is the async counterpart of context_with_sample_data.
    """
    provider = AsyncMockProvider(
        sample_values,
        sample_history,
        manifest=DEFAULT_MANIFEST
    )
    return Context([provider])


@pytest.fixture
def sample_manifest() -> Manifest:
    """Return a sample manifest with RSI and MACD indicators.

    This manifest includes parameter schemas (period, fast, slow)
    and attributes for use in validator tests.
    """
    return Manifest(
        indicators={
            'rsi': IndicatorSchema(
                parameters={
                    'period': ParameterSchema(
                        type='integer',
                        default=14,
                        min=1,
                        max=100
                    ),
                    'source': ParameterSchema(type='any', default='close')
                },
                attributes=['value', 'signal']
            ),
            'macd': IndicatorSchema(
                parameters={
                    'fast': ParameterSchema(type='integer', default=12, min=2),
                    'slow': ParameterSchema(type='integer', default=26, min=2),
                },
                attributes=['line', 'signal', 'histogram']
            )
        }
    )
