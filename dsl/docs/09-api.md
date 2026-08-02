# API Reference

This document provides a detailed API reference for the DSL.

---

## Core Modules

### `dsl.evaluate`

| Function | Signature | Description |
|:---------|:----------|:------------|
| `evaluate_dsl` | `(code: str, context: Context) -> bool` | Parses and evaluates a DSL expression synchronously. |
| `evaluate_dsl_async` | `(code: str, context: Context) -> Awaitable[bool]` | Parses and evaluates a DSL expression asynchronously. |

### `dsl.parser`

| Class/Method | Signature | Description |
|:-------------|:----------|:------------|
| `Parser` | | Recursive descent parser. |
| `.parse` | `(code: str) -> ASTNode` | Parses a DSL expression string and returns the root AST node. |
| `parse` (function) | `(code: str) -> ASTNode` | Convenience function, equivalent to `Parser().parse(code)`. |

### `dsl.interpreter`

| Class/Method | Signature | Description |
|:-------------|:----------|:------------|
| `Interpreter` | `(context: Context)` | Creates an interpreter with the given context. |
| `.visit` | `(node: ASTNode) -> bool` | Synchronously evaluates an AST node. |
| `.visit_async` | `(node: ASTNode) -> Awaitable[bool]` | Asynchronously evaluates an AST node. |

### `dsl.context`

| Class/Method | Signature | Description |
|:-------------|:----------|:------------|
| `Context` | `(providers: list[IndicatorProvider])` | Creates a context with a list of providers. |
| `.get_value` | `(indicator: str, params: dict, attributes: list, offset: int) -> float` | Synchronously retrieves an indicator value. |
| `.get_value_async` | `(indicator: str, params: dict, attributes: list, offset: int) -> Awaitable[float]` | Asynchronously retrieves an indicator value. |
| `.get_history` | `(indicator: str, params: dict, attributes: list, n: int) -> list[float]` | Synchronously retrieves historical values. |
| `.get_history_async` | `(indicator: str, params: dict, attributes: list, n: int) -> Awaitable[list[float]]` | Asynchronously retrieves historical values. |
| `.get_manifest` | `() -> Manifest` | Returns the aggregated manifest. |

### `dsl.ast` – AST Node Classes

All AST nodes are frozen dataclasses with `to_dict()` and `from_dict()` methods. The main types include:

| Node Type | Purpose |
|:----------|:--------|
| `Number` | Numeric literal |
| `Var` | Let‑bound variable reference |
| `IndicatorAccess` | Indicator without parameters (e.g., `close`) |
| `IndicatorWithParams` | Indicator with parameters (e.g., `rsi(period=14)`) |
| `Comparison` | Binary comparison (`<`, `>`, etc.) |
| `MultiComparison` | Chained comparison (`a < b <= c`) |
| `LogicalBinOp` | `and` / `or` |
| `LogicalNot` | `not` |
| `Let` | `let ... in ...` |
| `HistoricalAccess` | `expr[offset]` |
| `Rising` | `rising(expr, n)` |
| `Falling` | `falling(expr, n)` |
| Arithmetic nodes: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `UnaryMinus` |

---

## Providers

### `dsl.providers.base`

- `IndicatorProvider` – protocol for synchronous providers.
- `AsyncIndicatorProvider` – protocol for asynchronous providers.

### `dsl.providers.inprocess`

- `InProcessProvider` – calls a Python function to resolve indicators.

### `dsl.providers.http`

- `HTTPProvider` – synchronous HTTP client.
- `AsyncHTTPProvider` – asynchronous HTTP client.

### `dsl.providers.manifest`

- `Manifest` – container for manifest data.
- `IndicatorSchema` – schema for a single indicator.
- `ParameterSchema` – schema for a parameter.
- `ManifestValidator` – validates requests against the manifest.

---

## Exceptions

All exceptions inherit from `DSLError`.

- `ParseError` – syntax errors.
- `EvaluationError` – runtime errors.
- `ProviderError` – errors from providers.

---

## Utility Functions

### `dsl.ast.from_dict`

`from_dict(data: dict) -> ASTNode` – deserializes a dict to an AST node.

---

## Example: Using the API Directly

```python
from dsl.parser import Parser
from dsl.interpreter import Interpreter
from dsl.context import Context
from dsl.providers import InProcessProvider

# Setup
manifest = {"indicators": {"close": {"attributes": []}}}
def resolver(indicator, params, attrs, offset):
    return 150.0
provider = InProcessProvider(manifest, resolver)
ctx = Context([provider])

# Parse
parser = Parser()
ast = parser.parse("close > 100")

# Evaluate
interp = Interpreter(ctx)
result = interp.visit(ast)
print(result)   # True
```

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
