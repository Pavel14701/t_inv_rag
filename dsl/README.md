# What is the DSL?

The **DSL for Technical Indicators** is an embeddable, domain-specific language designed for writing trading conditions and rules in a concise, readable way. It lets you combine arithmetic, comparisons, logical operators, indicator calls (with parameters and attributes), historical offsets, and even `let` bindings into a single expression – all evaluated against live or historical market data.

Instead of writing verbose Python code to fetch indicators, compute values, and combine conditions, you write a string like:

```text
let r = rsi(period=14) in r > 70 and close[1] < close
```

And the DSL engine parses, validates, and evaluates it, returning `True` or `False`.

---

## Key Features

- **Expressive syntax** – close to natural language for trading rules.
- **Indicator support** – any indicator name, with named parameters and attributes.
- **Historical access** – `[offset]` to refer to past bars.
- **Rising / falling** – check monotonicity over a window.
- **Let bindings** – factor out sub‑expressions for clarity.
- **Short‑circuit evaluation** – efficient `and`/`or`.
- **Strict validation** – manifests describe available indicators, catching mistakes early.
- **Sync & async** – evaluate synchronously or asynchronously with the same AST.
- **Extensible** – plug in your own data providers (in‑process, HTTP, etc.).
- **Zero external dependencies** – core uses only the Python standard library.

---

## When to Use the DSL

- **Trading bots** – define entry/exit conditions.
- **Screening tools** – filter stocks or assets by technical patterns.
- **Backtesting** – encode strategies for historical simulation.
- **Alerting** – trigger notifications when conditions are met.
- **Configuration** – let non‑programmers (e.g., traders) write rules in a safe, validated language.

---

## Example Use Cases

### 1. Simple condition

```text
close > 200
```

### 2. Volume & RSI

```text
volume > 1000000 and rsi(period=14).value < 30
```

### 3. Trend following

```text
let ma = sma(period=20) in close > ma and rising(close, 5)
```

### 4. Momentum reversal

```text
close > high[1] and rsi > 70
```

---

## How It Works (High-Level)

1. **Lexing & Parsing** – the expression string is tokenized and parsed into an Abstract Syntax Tree (AST) using a recursive descent parser.
2. **Validation** – the AST is checked against a manifest (optional) to ensure that indicators, parameters, and attributes exist.
3. **Interpretation** – the interpreter walks the AST, resolving indicator values via a `Context` (or provider) and evaluating arithmetic, comparisons, and logic.
4. **Result** – a single boolean value is returned.

The design is modular: you can replace the data source, add new indicator providers, or extend the grammar.

---

### Table of Contents

- **[Overview](./README.md)** – what is the DSL and why use it
- **[Getting Started](docs/02-getting-started.md)** – installation and first run
- **[Language Syntax](docs/03-syntax.md)** – full grammar and operators
- **[Indicators & Parameters](docs/04-indicators.md)** – indicator names, parameters, attributes, historical offsets
- **[Let Expressions](docs/05-let.md)** – variable binding and scoping
- **[Context & Providers](docs/06-context.md)** – connecting your own data sources
- **[Examples](docs/07-examples.md)** – practical conditions and strategies
- **[Advanced Topics](docs/08-advanced.md)** – performance, debugging, error handling
- **[API Reference](docs/09-api.md)** – class and function documentation
