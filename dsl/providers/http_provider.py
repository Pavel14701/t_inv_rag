"""HTTP indicator provider with support for HTTP/2 and HTTP/3 using niquests.

This module provides synchronous and asynchronous HTTP providers
for communicating with remote indicator services.
"""

from typing import Any

import niquests

from ..exceptions import ProviderError
from .base import AsyncIndicatorProvider, IndicatorProvider


class HTTPProvider(IndicatorProvider):
    """Synchronous HTTP provider with HTTP/2 and HTTP/3 support using niquests.

    This provider communicates with a remote indicator service over HTTP/2
    or HTTP/3 (configurable).

    Attributes:
        base_url: Base URL of the remote service (e.g., http://localhost:8000).
        timeout: Request timeout in seconds.
        http_version: Which HTTP version to prefer: 'h2' (HTTP/2)
        or 'h3' (HTTP/3).Default 'h2'.

    """

    def __init__(
        self, base_url: str, timeout: float = 5.0, http_version: str = "h2"
    ) -> None:
        """Initialize the HTTP provider.

        Args:
            base_url: Base URL of the remote service.
            timeout: Request timeout in seconds.
            http_version: HTTP version to use: 'h2' or 'h3' (default 'h2').

        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http_version = http_version
        self._session: niquests.Session | None = None

    def _get_session(self) -> niquests.Session:
        """Get or create a session with the configured HTTP version."""
        if self._session is None:
            # Configure HTTP version via disable flags
            if self.http_version == "h2":
                disable_http1 = True
                disable_http2 = False
                disable_http3 = True
            elif self.http_version == "h3":
                disable_http1 = True
                disable_http2 = True
                disable_http3 = False
            else:
                raise ValueError(
                    f"Unsupported HTTP version: {self.http_version}.",
                    "Use 'h2' or 'h3'.",
                )

            self._session = niquests.Session(
                timeout=self.timeout,
                disable_http1=disable_http1,
                disable_http2=disable_http2,
                disable_http3=disable_http3,
            )
        return self._session

    def get_manifest(self) -> dict[str, Any]:
        """Fetch the manifest from the remote service.

        Returns:
            The manifest dictionary.

        Raises:
            ProviderError: If the request fails or the response is invalid.

        """
        try:
            session = self._get_session()
            resp = session.get(f"{self.base_url}/manifest")
            resp.raise_for_status()
            return resp.json()
        except niquests.RequestException as e:
            raise ProviderError(f"HTTP error fetching manifest: {e}") from e

    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Resolve an indicator value from the remote service.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attribute names.
            offset: Bar offset.

        Returns:
            The numeric value.

        Raises:
            ProviderError: If the request fails or the response
            indicates an error.

        """
        payload = {
            "indicator": indicator,
            "parameters": params,
            "attributes": attributes,
            "bar_offset": offset,
        }
        try:
            session = self._get_session()
            resp = session.post(f"{self.base_url}/resolve", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                raise ProviderError(data.get("error", "Unknown error"))
            return data["value"]
        except niquests.RequestException as e:
            raise ProviderError(f"HTTP error resolving indicator: {e}") from e

    def close(self) -> None:
        """Close the underlying session."""
        if self._session is not None:
            self._session.close()
            self._session = None


class AsyncHTTPProvider(AsyncIndicatorProvider):
    """Asynchronous HTTP provider for indicator resolution.

    This provider fetches indicator values from a remote service using
    asynchronous HTTP requests. It supports HTTP/2 and HTTP/3 for
    improved performance and reduced latency.

    Attributes:
        base_url: Base URL of the remote service (e.g., http://localhost:8000).
        timeout: Request timeout in seconds.
        http_version: HTTP version to prefer: 'h2' (HTTP/2) or 'h3' (HTTP/3).
            Default is 'h3' for async workflows.

    """

    def __init__(
        self, base_url: str, timeout: float = 5.0, http_version: str = "h3"
    ) -> None:
        """Initialize the asynchronous HTTP provider.

        Args:
            base_url: Base URL of the remote service.
            timeout: Request timeout in seconds.
            http_version: HTTP version: 'h2' or 'h3' (default 'h3').

        Raises:
            ValueError: If an unsupported HTTP version is provided.

        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http_version = http_version
        self._session: niquests.AsyncSession | None = None

    async def _get_session(self) -> niquests.AsyncSession:
        """Get or create an asynchronous session with the configured
        HTTP version.

        Returns:
            An `AsyncSession` instance configured for the requested
            HTTP version.

        Raises:
            ValueError: If the HTTP version is unsupported.

        """
        if self._session is None:
            # Configure HTTP version via disable flags
            if self.http_version == "h2":
                disable_http1, disable_http2, disable_http3 = True, False, True
            elif self.http_version == "h3":
                disable_http1, disable_http2, disable_http3 = True, True, False
            else:
                raise ValueError(
                    f"Unsupported HTTP version: {self.http_version}. "
                    "Use 'h2' or 'h3'."
                )

            self._session = niquests.AsyncSession(
                timeout=self.timeout,
                disable_http1=disable_http1,
                disable_http2=disable_http2,
                disable_http3=disable_http3,
            )
        return self._session

    async def get_manifest_async(self) -> dict[str, Any]:
        """Asynchronously fetch the manifest from the remote service.

        Returns:
            The manifest dictionary.

        Raises:
            ProviderError: If the request fails or the response is invalid.

        """
        try:
            session = await self._get_session()
            resp = await session.get(f"{self.base_url}/manifest")
            resp.raise_for_status()
            return resp.json()
        except niquests.RequestException as e:
            raise ProviderError(f"HTTP error fetching manifest: {e}") from e

    async def resolve_async(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Asynchronously resolve an indicator value from the remote service.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attribute names.
            offset: Bar offset.

        Returns:
            The numeric value.

        Raises:
            ProviderError: If the request fails or the response
            indicates an error.

        """
        payload = {
            "indicator": indicator,
            "parameters": params,
            "attributes": attributes,
            "bar_offset": offset,
        }
        try:
            session = await self._get_session()
            resp = await session.post(f"{self.base_url}/resolve", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                raise ProviderError(data.get("error", "Unknown error"))
            return data["value"]
        except niquests.RequestException as e:
            raise ProviderError(f"HTTP error resolving indicator: {e}") from e

    async def close(self) -> None:
        """Close the underlying asynchronous session.

        This method should be called when the provider is no longer needed
        to release network resources.
        """
        if self._session is not None:
            await self._session.close()
            self._session = None
