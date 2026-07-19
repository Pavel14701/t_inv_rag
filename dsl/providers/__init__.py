"""Indicator providers package.

This package provides various implementations of indicator providers that
can be used with the DSL context to resolve indicator values.

Available providers:
    - InProcessProvider: Direct in-process resolution using a Python callable.
        Useful for integrating custom indicators without network overhead.

    - HTTPProvider: Synchronous HTTP provider with HTTP/2 and HTTP/3 support
        using the niquests library. Connects to a remote indicator service.

    - AsyncHTTPProvider: Asynchronous version of HTTPProvider for
        async workflows. Can be used independently with asyncio.

Each provider must implement the IndicatorProvider interface, which requires
`get_manifest()` and `resolve()` methods.

Example:
    >>> from dsl.providers import InProcessProvider, HTTPProvider
    >>> from dsl.context import Context
    >>>
    >>> # In-process provider
    >>> def my_resolver(indicator, params, attributes, offset):
    ...     if indicator == 'rsi':
    ...         return 25.0
    ...     return 50.0
    >>> manifest = {'indicators': {'rsi': {'attributes': ['value']}}}
    >>> provider = InProcessProvider(manifest, my_resolver)
    >>> context = Context([provider])
    >>>
    >>> # HTTP provider
    >>> http_provider = HTTPProvider('http://localhost:8000', http_version='h2')
    >>> context2 = Context([http_provider])

"""  # noqa: E501

from .base import IndicatorProvider
from .manifest import (
    Manifest,
    ManifestValidator,
    ParameterSchema,
    IndicatorSchema,
)
from .in_process import InProcessProvider
from .http_provider import HTTPProvider, AsyncHTTPProvider

__all__ = (
    'IndicatorProvider',
    'Manifest',
    'ManifestValidator',
    'ParameterSchema',
    'IndicatorSchema',
    'InProcessProvider',
    'HTTPProvider',
    'AsyncHTTPProvider',
)
