# Advanced Topics

This document covers performance considerations, debugging, error handling, and how to extend the DSL.

---

## Performance

### Expression Complexity

The interpreter is a simple tree walker. Its performance is linear in the number of AST nodes. For typical trading rules (a few dozen nodes), evaluation is sub‑millisecond.

### Indicator Resolution Overhead

The dominant cost is usually the indicator resolution – fetching data from your database, API, or local storage. To improve performance:

- **Cache** indicator values for the same offset within a single evaluation. The context does not cache by default; you can implement caching inside your provider.
- **Batch history retrieval**: Override `resolve_history` (or `resolve_history_async`) to fetch multiple values in one call. The default implementation calls `resolve` for each offset, which can be slow for large `n`.
- **Use `let` to reuse** expensive calculations.

### Asynchronous Evaluation

If your providers are I/O‑bound (HTTP, database), use the async interface. This allows concurrent resolution of multiple indicators (if you use `asyncio.gather` or similar). However, the interpreter itself evaluates sequentially; you would need to implement parallel resolution at the provider level.

---

## Debugging

### Print AST

For debugging, you can serialize the AST to JSON:

```python
from dsl.parser import Parser

ast = Parser().parse("close > 100")
print(ast.to_dict())   # outputs a dict
```

### Verbose Errors

The parser and interpreter raise exceptions with detailed messages, including the position of the error (line/column) for parse errors, and the indicator name for resolution errors.

### Logging

You can add logging to your provider to trace which indicators are being requested and with what parameters.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Error Handling

### Exception Hierarchy

All DSL‑specific exceptions inherit from `DSLError`:

- `ParseError` – syntax error in the expression.
- `EvaluationError` – runtime error (e.g., division by zero, unknown indicator, type mismatch).
- `ProviderError` – error from a provider (e.g., network failure, missing data).

### Common Errors and Solutions

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `ParseError: Unexpected token` | Missing spaces around operators, unbalanced parentheses, or invalid character. | Check syntax; ensure spaces around operators; balance parentheses. |
| `EvaluationError: Division by zero` | Expression divides by zero. | Guard the division with a condition, e.g., `denominator != 0 and numerator / denominator ...` |
| `EvaluationError: Unknown indicator` | The indicator name is not known by the manifest or the provider. | Check the indicator name; update the manifest or the provider. |
| `EvaluationError: Invalid parameter` | The parameter name or value does not match the manifest. | Check parameter names and types; ensure they are within allowed ranges. |
| `ProviderError: No provider found` | No provider can resolve the indicator. | Add a provider that handles this indicator; check provider order. |
| `ValueError` (from validation) | The request failed manifest validation. | Update the manifest or correct the request. |

---

## Extending the DSL

### Adding New Functions

To add a new function (e.g., `cross(expr1, expr2)`), you would need to:

1. Extend the grammar in the parser.
2. Add a new AST node class.
3. Implement the interpretation logic in the interpreter.

This is an advanced topic that requires modifying the core code. For most users, the existing set of functions is sufficient.

### Custom Providers

You can easily add new providers by implementing the `IndicatorProvider` or `AsyncIndicatorProvider` interface. This is the recommended way to connect the DSL to your data.

### Custom Validation

You can extend the manifest validation by subclassing `ManifestValidator` and overriding the `validate` method.

---

## Security Considerations

### Expression Injection

Since the DSL is expression‑based and does not allow arbitrary code execution, the risk of injection is low. However, ensure that the context providers do not expose sensitive operations or data that could be abused.

### Timeouts

When using HTTP providers, set reasonable timeouts to avoid hanging the interpreter.

### Resource Limits

Consider limiting the complexity of expressions (e.g., maximum nesting depth, maximum number of indicator calls) to prevent denial of service. The current implementation does not enforce such limits.

---

## Testing

The DSL comes with a comprehensive test suite. You can run it with:

```bash
pytest tests/
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
