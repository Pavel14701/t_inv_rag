"""Unit tests for provider implementations.

This module tests the InProcessProvider and HTTP providers.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from ..providers import AsyncIndicatorProvider, InProcessProvider
from ..providers.base import IndicatorProvider


@pytest.mark.unit
@pytest.mark.provider
def test_inprocess_provider_get_manifest() -> None:
    """Test that InProcessProvider returns the manifest correctly."""
    manifest: dict[str, Any] = {"indicators": {"close": {"attributes": []}}}
    resolver = Mock()
    provider = InProcessProvider(manifest, resolver)
    assert provider.get_manifest() == manifest


@pytest.mark.unit
@pytest.mark.provider
def test_inprocess_provider_resolve() -> None:
    """Test that InProcessProvider calls the resolver
    with correct arguments.
    """
    manifest: dict[str, Any] = {"indicators": {}}
    resolver = Mock(return_value=42.0)
    provider = InProcessProvider(manifest, resolver)
    result = provider.resolve("close", {"param": 1}, ["attr"], 0)
    assert result == 42.0
    resolver.assert_called_once_with("close", {"param": 1}, ["attr"], 0)


@pytest.mark.unit
@pytest.mark.provider
def test_inprocess_provider_resolve_raises() -> None:
    """Test that InProcessProvider propagates exceptions from resolver."""
    manifest: dict[str, Any] = {"indicators": {}}
    resolver = Mock(side_effect=ValueError("test error"))
    provider = InProcessProvider(manifest, resolver)
    with pytest.raises(ValueError, match="test error"):
        provider.resolve("close", {}, [], 0)


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.skip(reason="Requires network or mocking")
def test_http_provider_get_manifest() -> None:
    """Test HTTPProvider get_manifest makes correct request."""
    # This test would require mocking niquests or running a test server


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.skip(reason="Requires network or mocking")
def test_http_provider_resolve() -> None:
    """Test HTTPProvider resolve makes correct request."""


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.skip(reason="Requires network or mocking")
async def test_async_http_provider_get_manifest_async() -> None:
    """Test AsyncHTTPProvider get_manifest_async makes correct request."""


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.skip(reason="Requires network or mocking")
async def test_async_http_provider_resolve_async() -> None:
    """Test AsyncHTTPProvider resolve_async makes correct request."""


@pytest.mark.unit
@pytest.mark.provider
def test_indicator_provider_abstract() -> None:
    """Test that IndicatorProvider is abstract and cannot be
    instantiated directly.
    """
    with pytest.raises(TypeError):
        IndicatorProvider()  # type: ignore


@pytest.mark.unit
@pytest.mark.provider
def test_async_indicator_provider_abstract() -> None:
    """Test that AsyncIndicatorProvider is abstract and cannot be instantiated
    directly.
    """
    with pytest.raises(TypeError):
        AsyncIndicatorProvider()  # type: ignore
