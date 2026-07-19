"""Context for evaluating indicators with manifest-based validation."""

from typing import Any

from .providers import (
    IndicatorProvider,
    Manifest,
    ManifestValidator,
    IndicatorSchema
)
from .exceptions import ProviderError


class Context:
    """Context that orchestrates indicator providers and validates requests.

    It aggregates manifests from all registered providers, validates each
    indicator request against the combined manifest, and delegates resolution
    to the appropriate provider.

    Attributes:
        providers: List of registered indicator providers.

    """

    def __init__(self, providers: list[IndicatorProvider]) -> None:
        """Initialize the context with a list of providers.

        Args:
            providers: List of IndicatorProvider instances.

        """
        self.providers = providers
        # Normalize and cache per-provider manifests
        self._provider_manifests: dict[IndicatorProvider, Manifest] = {
            provider: self._normalize_manifest(provider.get_manifest())
            for provider in self.providers
        }
        self._manifest = self._build_manifest()
        self._validator = ManifestValidator(self._manifest)

    def _normalize_manifest(
        self,
        manifest_data: dict[str, Any] | Manifest
    ) -> Manifest:
        """Convert provider manifest data to a Manifest object.

        Args:
            manifest_data: Either a dict or a Manifest instance.

        Returns:
            Normalized Manifest object.

        Raises:
            TypeError: If the manifest data is neither dict nor Manifest.

        """
        if isinstance(manifest_data, Manifest):
            return manifest_data
        if isinstance(manifest_data, dict):
            return Manifest.from_dict(manifest_data)
        raise TypeError('Provider manifest must be dict or Manifest')

    def _build_manifest(self) -> Manifest:
        """Aggregate manifests from all providers into a single Manifest.

        Returns:
            A combined Manifest object containing all indicators
            from all providers.

        """
        all_indicators: dict[str, IndicatorSchema] = {}
        for manifest in self._provider_manifests.values():
            all_indicators |= manifest.indicators
        return Manifest(indicators=all_indicators)

    def get_value(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int = 0,
    ) -> float:
        """Retrieve the current value of an indicator.

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
        if errors := self._validator.validate(indicator, params, attributes):
            raise ValueError(f'Validation errors: {", ".join(errors)}')
        for provider, manifest in self._provider_manifests.items():
            if indicator in manifest.indicators:
                try:
                    return provider.resolve(
                        indicator, params, attributes, offset
                    )
                except ProviderError:
                    continue

        raise ProviderError(f"No provider found for indicator '{indicator}'")

    def get_history(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        n: int,
    ) -> list[float]:
        """Retrieve historical values of an indicator for the last n bars.

        This default implementation calls get_value() for each offset.
        Subclasses may override this for optimised batch retrieval.

        Args:
            indicator: Name of the indicator.
            params: Dictionary of parameter names to values.
            attributes: List of attribute names to access.
            n: Number of bars to retrieve (including current).

        Returns:
            A list of values ordered from oldest to
            newest (index 0 = n-1 bars ago).

        """
        result: list[float] = []
        result.extend(
            self.get_value(indicator, params, attributes, offset)
            for offset in range(n - 1, -1, -1)
        )
        return result

    def get_manifest(self) -> Manifest:
        """Return the combined manifest of all registered providers.

        Returns:
            The aggregated Manifest object.

        """
        return self._manifest
