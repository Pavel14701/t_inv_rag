"""Base provider interface for indicators."""

from abc import ABC, abstractmethod
from typing import Any

from .manifest import Manifest


class IndicatorProvider(ABC):
    """Abstract base class for indicator providers."""

    @abstractmethod
    def get_manifest(self) -> dict[str, Any] | Manifest:
        """Return the manifest of available indicators.

        Returns:
            Either a dictionary in the standard manifest format or a
            Manifest object.

        """
        raise NotImplementedError

    @abstractmethod
    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Compute the value of the given indicator.

        Args:
            indicator: Name of the indicator.
            params: Parameter dictionary.
            attributes: List of attribute names to access.
            offset: Bar offset (0 = current).

        Returns:
            The numeric value of the indicator.

        """
        raise NotImplementedError

    def resolve_history(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        n: int,
    ) -> list[float]:
        """Retrieve historical values for the last n bars.

        This default implementation retrieves values sequentially by calling
        `resolve` for each offset, starting from the oldest (n-1) to the
        current (0). Providers that can fetch multiple bars more efficiently
        (e.g., in a single batch request) should override this method to
        improve performance.

        Args:
            indicator: Name of the indicator.
            params: Dictionary of parameter names to values.
            attributes: List of attribute names to access.
            n: Number of bars to retrieve
                (including current bar). Must be >= 1.

        Returns:
            A list of numeric values ordered from oldest to newest.
            The element at index 0 corresponds to the bar (n-1) bars ago,
            and the last element corresponds to the current bar (offset 0).

        Raises:
            ValueError: If `n` is less than 1.
            ProviderError: If resolution fails for any of the offsets
                (implementations may propagate errors from `resolve`).

        Example:
            >>> provider = MyProvider()
            >>> history = provider.resolve_history("close", {}, [], 3)
            >>> # returns [close at offset 2, close at offset 1,
                close at offset 0]

        """
        if n <= 1:
            raise ValueError("n must be at least 1")
        return [
            self.resolve(indicator, params, attributes, offset)
            for offset in range(n - 1, -1, -1)
        ]


class AsyncIndicatorProvider(ABC):
    """Abstract base class for asynchronous indicator providers.

    Async providers implement network, database, or other I/O-bound operations
    using async/await. They are used by the context's asynchronous methods
    (`get_value_async`, `get_history_async`) to resolve indicators without
    blocking the event loop.

    Providers may also choose to implement the synchronous interface
    (`IndicatorProvider`) for backwards compatibility, but this is
    not required.
    """

    @abstractmethod
    async def get_manifest_async(self) -> dict[str, Any] | Manifest:
        """Asynchronously retrieve the manifest of available indicators.

        The manifest describes all indicators provided by this provider, along
        with their parameters and attributes. It is used for validation
        before resolving values.

        Returns:
            Either a dictionary in the standard manifest format, or a
            `Manifest` object.

        Raises:
            ProviderError: If the manifest cannot be retrieved (e.g., network
                error, invalid response).

        Example:
            >>> provider = AsyncHTTPProvider("http://api.example.com")
            >>> manifest = await provider.get_manifest_async()
            >>> print(manifest.indicators.keys())
            dict_keys(['close', 'rsi', 'macd'])

        """
        raise NotImplementedError

    @abstractmethod
    async def resolve_async(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Asynchronously compute the value of the given indicator.

        This is the core method that resolves a single data point for a
        specific indicator, with given parameters, attributes, and bar offset.

        Args:
            indicator: Name of the indicator (e.g., "close", "rsi", "macd").
            params: Dictionary of parameter names to values.
                Example: `{"period": 14}`.
            attributes: List of attribute names to access.
                Example: `["value"]` for `rsi.value`, or `["line"]` for MACD.
            offset: Bar offset relative to the current bar.
                `0` = current bar, `1` = previous bar, etc.

        Returns:
            The numeric value of the indicator at the specified offset.

        Raises:
            ProviderError: If the indicator cannot be resolved (e.g., unknown
                indicator, invalid parameters, missing data).

        Example:
            >>> provider = AsyncHTTPProvider("http://api.example.com")
            >>> value = await provider.resolve_async("close", {}, [], 0)
            >>> print(value)
            150.25

        """
        raise NotImplementedError

    async def resolve_history_async(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        n: int,
    ) -> list[float]:
        """Asynchronously retrieve historical values for the last n bars.

        This method fetches the historical data points for the specified
        indicator over the last `n` bars, including the current bar.
        The default implementation calls `resolve_async` sequentially
        for each offset from `n-1` down to `0`.

        Providers that can fetch multiple bars more efficiently
        (e.g., in a single batch request) should override this method
        to improve performance and reduce latency.

        Args:
            indicator: Name of the indicator.
            params: Dictionary of parameter names to values.
            attributes: List of attribute names to access.
            n: Number of bars to retrieve, including the current bar.
                Must be >= 1.

        Returns:
            A list of numeric values ordered from oldest to newest.
            The element at index `0` corresponds to the bar `(n-1)` bars ago,
            and the last element (index `n-1`) corresponds to the current bar
            (offset 0).

        Raises:
            ValueError: If `n` is less than 1.
            ProviderError: If any resolution fails (implementations may
                propagate errors from `resolve_async` or
                handle them internally).

        Example:
            >>> provider = AsyncHTTPProvider("http://api.example.com")
            >>> history = await provider.resolve_history_async("close", {}, [], 3)
            >>> # returns [close at offset 2, close at offset 1, close at offset 0]
            >>> print(history)
            [148.0, 150.25, 152.5]

        """  # noqa: E501
        if n <= 1:
            raise ValueError("n must be at least 1")
        # Fetch values from oldest (n-1) to newest (0)
        return [
            await self.resolve_async(indicator, params, attributes, offset)
            for offset in range(n - 1, -1, -1)
        ]
