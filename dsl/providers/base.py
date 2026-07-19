"""Base provider interface for indicators."""

from abc import abstractmethod
from typing import Any, Protocol

from .manifest import Manifest


class IndicatorProvider(Protocol):
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
