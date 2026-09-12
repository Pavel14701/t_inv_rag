import asyncio

from .exceptions import DslValidationError, ProviderError
from .providers import (
    AsyncIndicatorProvider,
    IndicatorProvider,
    IndicatorSchema,
    Manifest,
    ManifestValidator,
)


class Context:
    """Hybrid context that orchestrates indicator providers and supports
    both sync and async resolution.

    It aggregates manifests from all registered providers, validates each
    request, and delegates resolution to the appropriate provider.
    Both synchronous and asynchronous methods are provided, so the same
    context can be used in either execution model.

    Attributes:
        providers: List of registered indicator providers.

    """

    def __init__(self, providers: list):
        """Initialize the context with a list of providers.

        Args:
            providers: List of IndicatorProvider instances. They can be
                synchronous, asynchronous, or hybrid.

        """
        self.providers = providers
        self._provider_manifests = {}
        for p in providers:
            manifest = p.get_manifest()
            if isinstance(manifest, dict):
                manifest = Manifest.from_dict(manifest)
            self._provider_manifests[p] = manifest
        self._manifest = self._build_manifest()
        self._validator = ManifestValidator(self._manifest)

    def _build_manifest(self) -> Manifest:
        """Aggregate manifests from all providers into a single Manifest.

        Returns:
            A combined Manifest object.

        """
        manifest: Manifest
        all_indicators: dict[str, IndicatorSchema] = {}
        for manifest in self._provider_manifests.values():
            all_indicators |= manifest.indicators
        return Manifest(indicators=all_indicators)

    def get_manifest(self) -> Manifest:
        """Return the combined manifest of all registered providers.

        Returns:
            The aggregated Manifest object.

        """
        return self._manifest

    def _validate(self, indicator, params, attributes):
        """Validate an indicator request against the manifest.

        Raises:
            DslValidationError: If validation fails (also a ValueError
                for backward compatibility; TZ-01 п.2.2).

        """
        if errors := self._validator.validate(indicator, params, attributes):
            raise DslValidationError(
                f"Validation errors: {', '.join(errors)}"
            )

    def _candidates(self, indicator: str) -> list:
        """Провайдеры, чей манифест содержит этот индикатор (TZ-01 п.2.1).

        Отбор по манифесту вместо ``getattr``-проверок: детерминировано,
        O(1), и позволяет отличить «провайдер не знает индикатор»
        (пропустить) от «ошибка данных на этом баре» (пробросить).

        """
        return [
            p
            for p in self.providers
            if indicator in self._provider_manifests[p].indicators
        ]

    def get_value(
        self,
        indicator: str,
        params: dict,
        attributes: list,
        offset: int = 0
    ) -> float:
        """Synchronously retrieve the current value of an indicator.

        Args:
            indicator: Name of the indicator.
            params: Dictionary of parameter names to values.
            attributes: List of attribute names to access.
            offset: Bar offset (0 = current, positive = historical).

        Returns:
            The numeric value of the indicator.

        Raises:
            ValueError: If validation fails.
            ProviderError: If no provider can resolve the indicator.

        """
        provider: IndicatorProvider
        self._validate(indicator, params, attributes)
        first_error: ProviderError | None = None
        for provider in self._candidates(indicator):
            try:
                return provider.resolve(
                    indicator,
                    params,
                    attributes,
                    offset
                )
            except ProviderError as exc:
                # Preserve the most specific error (e.g. warmup
                # unavailability) instead of masking it with a
                # generic "No provider found".
                if first_error is None:
                    first_error = exc
                continue
        if first_error is not None:
            raise first_error
        raise ProviderError(f"No provider found for indicator '{indicator}'")

    def get_history(
        self,
        indicator: str,
        params: dict,
        attributes: list,
        n: int
    ) -> list[float]:
        """Synchronously retrieve historical values for the last n bars.

        If a provider supports `resolve_history`, it will be used; otherwise
        it falls back to sequential calls to `get_value`.

        Args:
            indicator: Name of the indicator.
            params: Parameter dictionary.
            attributes: List of attribute names.
            n: Number of bars to retrieve (including current).

        Returns:
            A list of values ordered from
            oldest to newest (index 0 = n-1 bars ago).

        """
        provider: IndicatorProvider
        self._validate(indicator, params, attributes)
        first_error: ProviderError | None = None
        for provider in self._candidates(indicator):
            if getattr(provider, 'resolve_history', None) is not None:
                try:
                    return provider.resolve_history(
                        indicator,
                        params,
                        attributes,
                        n
                    )
                except ProviderError as exc:
                    if first_error is None:
                        first_error = exc
                    continue
        # fallback: sequential calls
        return [
            self.get_value(indicator, params, attributes, i)
            for i in range(n - 1, -1, -1)
        ]

    async def get_value_async(
        self,
        indicator: str,
        params: dict,
        attributes: list,
        offset: int = 0
    ) -> float:
        """Asynchronously retrieve the current value of an indicator.

        If a provider implements `resolve_async`, it will be awaited;
        otherwise, if it provides a synchronous `resolve`, it is run in
        a thread executor.

        Args:
            indicator: Name of the indicator.
            params: Parameter dictionary.
            attributes: List of attribute names.
            offset: Bar offset.

        Returns:
            The numeric value.

        Raises:
            ValueError: If validation fails.
            ProviderError: If no provider can resolve the indicator.

        """
        provider: AsyncIndicatorProvider | IndicatorProvider
        self._validate(indicator, params, attributes)
        first_error: ProviderError | None = None
        for provider in self._candidates(indicator):
            if getattr(provider, 'resolve_async', None) is not None:
                try:
                    got = await provider.resolve_async(  # type: ignore[union-attr]
                        indicator,
                        params,
                        attributes,
                        offset
                    )
                    return got
                except ProviderError as exc:
                    if first_error is None:
                        first_error = exc
                    continue
            elif getattr(provider, 'resolve', None) is not None:
                loop = asyncio.get_running_loop()
                try:
                    return await loop.run_in_executor(
                        None,
                        provider.resolve,  # type: ignore[union-attr]
                        indicator,
                        params,
                        attributes,
                        offset
                    )
                except ProviderError as exc:
                    if first_error is None:
                        first_error = exc
                    continue
        if first_error is not None:
            raise first_error
        raise ProviderError(f"No provider found for indicator '{indicator}'")

    async def get_history_async(
        self,
        indicator: str,
        params: dict,
        attributes: list,
        n: int
    ) -> list[float]:
        """Asynchronously retrieve historical values for the last n bars.

        If a provider implements `resolve_history_async`, it will be used;
        otherwise it falls back to sequential calls to `get_value_async`.

        Args:
            indicator: Name of the indicator.
            params: Parameter dictionary.
            attributes: List of attribute names.
            n: Number of bars to retrieve (including current).

        Returns:
            A list of values ordered from oldest to newest.

        """
        provider: AsyncIndicatorProvider
        self._validate(indicator, params, attributes)
        first_error: ProviderError | None = None
        for provider in self._candidates(indicator):
            if getattr(provider, 'resolve_history_async', None) is not None:
                try:
                    return await provider.resolve_history_async(
                        indicator,
                        params,
                        attributes,
                        n
                    )
                except ProviderError as exc:
                    if first_error is None:
                        first_error = exc
                    continue
        # fallback: sequential calls
        return [
            await self.get_value_async(indicator, params, attributes, i)
            for i in range(n - 1, -1, -1)
        ]
