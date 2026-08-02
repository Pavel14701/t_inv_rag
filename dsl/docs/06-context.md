# Context and Providers

The DSL separates **logic** (the expression) from **data** (indicator values). The `Context` and `Provider` interfaces handle all data retrieval.

---

## Overview

- **`Context`** – orchestrates providers, validates requests against a manifest, and delegates to the appropriate provider.
- **`IndicatorProvider`** – resolves individual indicator values (synchronous).
- **`AsyncIndicatorProvider`** – resolves indicator values asynchronously.

You can have multiple providers; the context will try them in order until one succeeds.

---

## The Context Class

### Basic Usage

```python
from dsl.context import Context
from dsl.providers import InProcessProvider

# Create a provider
provider = InProcessProvider(manifest, resolver_func)
ctx = Context([provider])
```

### Methods

- `get_value(indicator, params, attributes, offset)` – synchronous resolution.
- `get_value_async(indicator, params, attributes, offset)` – asynchronous resolution.
- `get_history(indicator, params, attributes, n)` – synchronous history retrieval.
- `get_history_async(indicator, params, attributes, n)` – asynchronous history retrieval.
- `get_manifest()` – returns the aggregated manifest from all providers.

### Validation

The context validates all requests against a **manifest** (if any provider provides one). Validation checks:

- Indicator existence.
- Parameter names, types, and ranges.
- Attribute names.

If validation fails, a `ValueError` is raised.

---

## Providers

### `IndicatorProvider` Protocol

```python
class IndicatorProvider(ABC):
    @abstractmethod
    def get_manifest(self) -> dict | Manifest:
        """Return the manifest of available indicators."""
        raise NotImplementedError

    @abstractmethod
    def resolve(self, indicator, params, attributes, offset) -> float:
        """Compute the value of the given indicator."""
        raise NotImplementedError
```

### `AsyncIndicatorProvider` Protocol

Adds `async` versions:

```python
class AsyncIndicatorProvider(ABC):
    @abstractmethod
    async def get_manifest_async(self) -> dict | Manifest: ...
    @abstractmethod
    async def resolve_async(self, indicator, params, attributes, offset) -> float: ...
```

### Provided Providers

| Provider | Description |
|:---------|:------------|
| `InProcessProvider` | Calls a Python function to resolve indicators. Fast and simple. |
| `HTTPProvider` | Synchronous HTTP client (uses `niquests`). Supports HTTP/2 and HTTP/3. |
| `AsyncHTTPProvider` | Asynchronous HTTP client (uses `niquests` async). Supports HTTP/2 and HTTP/3. |

#### Example: `InProcessProvider`

```python
from dsl.providers import InProcessProvider

manifest = {"indicators": {"close": {"attributes": []}}}
def resolver(indicator, params, attributes, offset):
    if indicator == "close":
        return 150.0
    raise ValueError("Unknown indicator")

provider = InProcessProvider(manifest, resolver)
ctx = Context([provider])
```

#### Example: `HTTPProvider`

```python
from dsl.providers import HTTPProvider

provider = HTTPProvider("http://localhost:8000", timeout=5.0, http_version="h2")
ctx = Context([provider])
```

---

## Manifests

A **manifest** describes available indicators, their parameters, and attributes. It is used for validation.

### Structure

```python
manifest = {
    "indicators": {
        "rsi": {
            "attributes": ["value", "signal"],
            "parameters": {
                "period": {"type": "integer", "default": 14}
            }
        }
    }
}
```

### Using the Manifest Classes

```python
from dsl.providers.manifest import Manifest, IndicatorSchema, ParameterSchema

manifest = Manifest(
    indicators={
        "rsi": IndicatorSchema(
            parameters={"period": ParameterSchema(type="integer", default=14)},
            attributes=["value", "signal"]
        )
    }
)
```

### Validator

`ManifestValidator` validates requests against a manifest.

```python
validator = ManifestValidator(manifest)
errors = validator.validate("rsi", {"period": 14}, ["value"])
if errors:
    print(errors)  # []
```

---

## Multiple Providers

The context tries providers in order. If the first provider fails (raises `ProviderError`), it tries the next one.

```python
from dsl.providers import HTTPProvider, InProcessProvider

ctx = Context([
    HTTPProvider("http://primary.service"),   # first attempt
    HTTPProvider("http://backup.service"),    # fallback
    InProcessProvider(local_manifest, local_resolver)  # last resort
])
```

This gives you **fault tolerance** and allows you to combine different data sources.

---

## Implementing a Custom Provider

1. Subclass `IndicatorProvider` (or `AsyncIndicatorProvider`).
2. Implement `get_manifest()`.
3. Implement `resolve()` (or `resolve_async()`).
4. Optionally override `resolve_history()` for batch retrieval.

```python
from dsl.providers.base import IndicatorProvider

class MyProvider(IndicatorProvider):
    def get_manifest(self):
        return {"indicators": {"my_indicator": {"attributes": []}}}

    def resolve(self, indicator, params, attributes, offset):
        if indicator == "my_indicator":
            return self._fetch_value(offset)
        raise ProviderError("Unknown")

    def resolve_history(self, indicator, params, attributes, n):
        if indicator == "my_indicator":
            return self._fetch_history(n)
        raise ProviderError("Unknown")
```

---

## Async Workflow

1. Use `AsyncIndicatorProvider` implementations.
2. Use `evaluate_dsl_async` or `Interpreter.visit_async`.
3. The context will call `get_value_async` and `get_history_async`.

```python
result = await evaluate_dsl_async("close > 100", ctx)
```

---

## Summary

|  Concept |  Description |
|:---------|:-------------|
| `Context` | Orchestrates providers and validation |
| `IndicatorProvider` | Synchronous data source |
| `AsyncIndicatorProvider` | Asynchronous data source |
| `Manifest` | Describes available indicators (for validation) |
| `ManifestValidator` | Validates requests against a manifest |
| Multiple providers | Fallback mechanism |

---

### Table of Contents

- **[Overview](../README.md)** – what is the DSL and why use it
- **[Getting Started](./02-getting-started.md)** – installation and first run
- **[Language Syntax](./03-syntax.md)** – full grammar and operators
- **[Indicators & Parameters](./04-indicators.md)** – indicator names, parameters, attributes, historical offsets
- **[Let Expressions](./05-let.md)** – variable binding and scoping
- **[Context & Providers](./06-context.md)** – connecting your own data sources
- **[Examples](./07-examples.md)** – practical conditions and strategies
- **[Advanced Topics](./08-advanced.md)** – performance, debugging, error handling
- **[API Reference](./09-api.md)** – class and function documentation
