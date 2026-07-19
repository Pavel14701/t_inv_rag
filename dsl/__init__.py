"""DSL for describing trading signals.

This package provides a domain-specific language for defining trading
conditions based on indicator values. It includes parsing, AST,
interpretation, and a provider system for integrating various data sources.

Key components:
    - evaluate_dsl(): Main entry point for evaluating expressions.
    - Context: Orchestrates providers and validates indicator requests.
    - IndicatorProvider: Base class for all indicator providers.
    - Manifest, ManifestValidator: Schema definitions and validation.
    - InProcessProvider: In-process indicator resolution.
    - HTTPProvider, AsyncHTTPProvider: Remote HTTP providers with HTTP/2/3.
    - DSLError, ParseError, EvaluationError, ProviderError: Exception
        hierarchy.

Example:
    >>> from dsl import evaluate_dsl, Context, InProcessProvider
    >>>
    >>> def resolver(indicator, params, attributes, offset):
    ...     if indicator == 'rsi':
    ...         return 25.0
    ...     return 50.0
    >>>
    >>> manifest = {'indicators': {'rsi': {'attributes': ['value']}}}
    >>> provider = InProcessProvider(manifest, resolver)
    >>> context = Context([provider])
    >>> result = evaluate_dsl('rsi.value < 30', context)
    >>> print(result)  # True

"""

from .evaluate import evaluate_dsl
from .context import Context
from .providers import (
    IndicatorProvider,
    Manifest,
    ManifestValidator,
    ParameterSchema,
    IndicatorSchema,
    InProcessProvider,
    HTTPProvider,
    AsyncHTTPProvider,
)
from .exceptions import DSLError, ParseError, EvaluationError, ProviderError

__all__ = (
    'evaluate_dsl',
    'Context',
    'IndicatorProvider',
    'Manifest',
    'ManifestValidator',
    'ParameterSchema',
    'IndicatorSchema',
    'InProcessProvider',
    'HTTPProvider',
    'AsyncHTTPProvider',
    'DSLError',
    'ParseError',
    'EvaluationError',
    'ProviderError',
)
