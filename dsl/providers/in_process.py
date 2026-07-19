"""In-process indicator provider."""

from collections.abc import Callable
from typing import Any

from .base import IndicatorProvider


class InProcessProvider(IndicatorProvider):
    """In-process provider that calls a resolver function directly.

    This provider is useful for integrating indicators implemented in Python
    without network overhead. The resolver function must have the signature
    (indicator, params, attributes, offset) -> float.

    Attributes:
        manifest: Dictionary describing available indicators and their schemas.
        resolver: Callable that resolves indicator values.

    """

    def __init__(
        self,
        manifest: dict[str, Any],
        resolver_func: Callable[[str, dict[str, Any], list[str], int], float],
    ) -> None:
        """Initialize the in-process provider.

        Args:
            manifest: Manifest dictionary.
            resolver_func: Function to resolve indicator values.

        """
        self.manifest = manifest
        self.resolver = resolver_func

    def get_manifest(self) -> dict[str, Any]:
        """Return the manifest of available indicators.

        Returns:
            The manifest dictionary.

        """
        return self.manifest

    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Resolve an indicator value using the provided resolver function.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attribute names.
            offset: Bar offset.

        Returns:
            The numeric value.

        """
        return self.resolver(indicator, params, attributes, offset)
