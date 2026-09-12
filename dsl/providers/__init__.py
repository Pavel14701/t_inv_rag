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

from .base import AsyncIndicatorProvider, IndicatorProvider
from .http_provider import AsyncHTTPProvider, HTTPProvider
from .in_process import InProcessProvider
from .manifest import (
    IndicatorSchema,
    Manifest,
    ManifestValidator,
    ParameterSchema,
)


__all__ = (
    'AsyncHTTPProvider',
    'AsyncIndicatorProvider',
    'HTTPProvider',
    'InProcessProvider',
    'IndicatorProvider',
    'IndicatorSchema',
    'Manifest',
    'ManifestValidator',
    'ParameterSchema',
)
