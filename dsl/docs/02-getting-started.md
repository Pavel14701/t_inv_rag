# Getting Started with the DSL

This guide walks you through installing the DSL, writing your first expression, and integrating it into a Python project.

---

## Installation

The DSL requires Python 3.10 or later. It is distributed as a Python package. Use `uv` (recommended) or `pip`:

### Using `uv` (fast and modern)

```bash
uv add "git+https://github.com/your/repo.git"   # replace with actual URL
```

### Using `pip`

```bash
pip install "git+https://github.com/your/repo.git"
```

### Development install (if you plan to modify the DSL)

```bash
git clone https://github.com/your/repo.git
cd repo
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .
```

---

## Your First Expression

Let's evaluate a simple condition: `close > 100`.

### Step 1: Define a Context

The context is responsible for providing indicator values. For a quick test, we'll create a dummy context.

```python
from dsl.context import Context

class DummyContext(Context):
    def get_value(self, indicator, params, attributes, offset):
        if indicator == "close":
            return 150.0   # pretend the current close is 150
        raise ValueError(f"Unknown indicator: {indicator}")

    def get_history(self, indicator, params, attributes, n):
        # Not needed for this simple example, but must be implemented
        return [120, 130, 140, 150]
```

### Step 2: Parse and Evaluate

```python
from dsl.parser import Parser
from dsl.interpreter import Interpreter

code = "close > 100"
parser = Parser()
ast = parser.parse(code)          # parse into an AST
ctx = DummyContext()
interp = Interpreter(ctx)
result = interp.visit(ast)        # returns True or False

print(result)   # True, because 150 > 100
```

### Step 3: Use the High‑Level `evaluate_dsl` Function

For convenience, there is a one‑liner:

```python
from dsl.evaluate import evaluate_dsl

result = evaluate_dsl("close > 100", ctx)
print(result)   # True
```

---

## Command‑Line Interface

The DSL includes a simple CLI to test expressions quickly.

```bash
echo "close > 100" | uv run -m dsl.evaluate
```

Or with a file:

```bash
uv run -m dsl.evaluate -f my_rule.dsl
```

The CLI reads from stdin or a file, parses the expression, and prints `True` or `False`. It expects a context to be provided – by default it uses a mock context that returns fixed values. For real usage, you would implement a custom context and pass it via the `--context` option (or environment variable).

---

## Basic Workflow in Your Application

1. **Create a Context** – implement `get_value` (and optionally `get_history` for `rising`/`falling`). You can also use one of the built‑in providers (`InProcessProvider`, `HTTPProvider`, etc.) – see [Context & Providers](./06-context.md).

2. **Write your DSL expression** – as a string.

3. **Parse and evaluate** – using `evaluate_dsl` for synchronous execution, or `evaluate_dsl_async` for asynchronous.

```python
from dsl.evaluate import evaluate_dsl

# synchronously
is_condition_met = evaluate_dsl("rsi(period=14).value < 30 and close > 200", ctx)

# asynchronously (inside an async function)
is_condition_met = await evaluate_dsl_async("...", ctx)
```

---

## Troubleshooting

### `ParseError: Unexpected token`

- Check for missing spaces around operators (e.g., `close>100` → `close > 100`).
- Ensure parentheses, brackets, and commas are balanced.

### `EvaluationError: Unknown indicator`

- The context does not recognize the indicator name. Check your `get_value` implementation or the manifest.

### `ProviderError: No provider found`

- Your context has no provider that can resolve the given indicator. If using `HTTPProvider`, make sure the service is running.

### Division by zero

- The DSL raises `EvaluationError` if a division or modulo by zero occurs. Avoid such expressions, or guard them with a condition.

---

For a complete list of errors, see [Advanced Topics](./08-advanced.md#error-handling).

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
