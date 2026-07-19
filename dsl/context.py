"""Context for evaluating indicators with manifest-based validation."""

from typing import Any, cast

from .exceptions import ProviderError
from .providers import (
    IndicatorProvider,
    Manifest,
    ManifestValidator,
    ParameterSchema,
    IndicatorSchema
)


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
        self._manifest = self._build_manifest()
        self._validator = ManifestValidator(self._manifest)

    def _build_manifest(self) -> Manifest:
        """Aggregate manifests from all providers into a single Manifest.

        Returns:
            A combined Manifest object containing all indicators from
            all providers.

        Raises:
            TypeError: If any provider returns an unsupported manifest type.

        """
        all_indicators: dict[str, IndicatorSchema] = {}
        for provider in self.providers:
            manifest_data = provider.get_manifest()
            if isinstance(manifest_data, dict):
                indicators_data = cast(
                    dict[str, Any],
                    manifest_data.get('indicators', {})
                )
                for name, raw_schema in indicators_data.items():
                    schema_data = cast(dict[str, Any], raw_schema)
                    params_data = cast(
                        dict[str, Any],
                        schema_data.get('parameters', {})
                    )
                    pdata: dict[str, Any]  # noqa: F842
                    params: dict[str, ParameterSchema] = {
                        pname: ParameterSchema(
                            type=pdata.get('type', 'any'),
                            default=pdata.get('default'),
                            min=pdata.get('min'),
                            max=pdata.get('max'),
                        )
                        for pname, pdata in params_data.items()
                    }
                    all_indicators[name] = IndicatorSchema(
                        parameters=params,
                        attributes=schema_data.get('attributes', []),
                    )
            elif isinstance(manifest_data, Manifest):
                for name, schema in manifest_data.indicators.items():
                    all_indicators[name] = schema
            else:
                raise TypeError('Provider manifest must be dict or Manifest')
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
            ValueError: If validation fails or no provider handles
            the indicator.

        """
        if errors := self._validator.validate(indicator, params, attributes):
            raise ValueError(f'Validation errors: {", ".join(errors)}')
        for provider in self.providers:
            try:
                prov_manifest = provider.get_manifest()
                if isinstance(prov_manifest, dict):
                    if indicator in prov_manifest.get('indicators', {}):
                        return provider.resolve(
                            indicator, params, attributes, offset
                        )
                elif isinstance(prov_manifest, Manifest):
                    if indicator in prov_manifest.indicators:
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
            A list of values ordered from oldest
            to newest (index 0 = n-1 bars ago).

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
