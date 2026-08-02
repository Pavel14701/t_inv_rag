# Indicators, Parameters, Attributes, and Offsets

This document explains how to use technical indicators in the DSL – from simple names to complex calls with parameters, attributes, and historical offsets.

---

## Indicator Name

Any identifier that is **not** a reserved keyword (`let`, `in`, `and`, `or`, `not`, `rising`, `falling`) can be an indicator name.

Examples:

```text
close
volume
rsi
macd
sma
```

The DSL does **not** know which indicators exist in advance – that is up to the `Context` implementation and the optional manifest.

---

## Named Parameters

Indicators often require parameters (e.g., `period`, `fast`, `slow`). You can pass them by name in parentheses:

```text
rsi(period=14)
macd(fast=12, slow=26)
sma(period=20)
```

Parameter names are identifiers, and their values can be **any expression** (not just literals):

```text
rsi(period=14 + 1)                     # period = 15
sma(period=10 + close > 100 ? 5 : 20)  # conditional not supported, but you can use arithmetic/logic
```

In practice, parameters are usually numeric literals or simple expressions.

---

## Attributes

Indicators may return multiple values (e.g., `rsi` returns a single value, `macd` may return `line`, `signal`, and `histogram`). Use dot notation to access them:

```text
rsi.value
macd.line
macd.signal
macd.histogram
```

Multiple attributes are allowed: `a.b.c` is parsed as `indicator='a'` with `attributes=['b','c']`. However, typical usage is one attribute.

Combine parameters and attributes:

```text
rsi(period=14).value
macd(fast=12, slow=26).signal
```

---

## Historical Offset

Access past values by appending `[N]` where `N` is a non‑negative integer. `[0]` is the current bar, `[1]` is the previous bar, etc.

```text
close[1]        # previous close
rsi(14).value[2]   # RSI value two bars ago
high[0]         # current high (same as high)
```

**Important restrictions:**

- Historical offset can only be applied to an **indicator access** (with or without parameters/attributes). It **cannot** be applied to arbitrary expressions:
  - `(close + 1)[1]` → `ParseError`
  - `(rsi.value)[1]` → `ParseError`
- The offset must be a **literal integer** – expressions are not allowed inside brackets.

---

## Rising and Falling Functions

These functions check if an indicator has been **strictly increasing** or **strictly decreasing** over the last `n` bars.

```text
rising(expr, n)
falling(expr, n)
```

### Requirements

- `expr` **must** be an indicator access (optionally with parameters and attributes).
- `expr` **cannot** be an arithmetic expression.
- `expr` **cannot** be a let‑bound variable.
- Any **historical offset** (`[offset]`) inside `expr` is **ignored**. To avoid confusion, do not use offsets inside `rising`/`falling`.

### Valid Examples

```text
rising(close, 3)
rising(rsi(period=14).value, 5)
falling(volume, 2)
```

### Invalid Examples

```text
rising(close + 1, 3)            # Error – expression not allowed
let x = close in rising(x, 3)   # Error – x is a number, not an indicator
rising(close[1], 3)             # Allowed but offset is ignored – avoid this
```

### Recommended Practice

Always pass the **bare indicator** (without offset) to `rising`/`falling`. If you need to compare shifted values, do that **outside** the function:

```text
rising(close, 3) and close[1] > close[2]
```

---

## Indicator Resolution and Manifest

When the interpreter encounters an indicator reference, it calls the context's `get_value` (or `get_value_async`). The context may validate the indicator name, parameters, and attributes against a **manifest**. The manifest describes what indicators are available, which parameters they accept (types, ranges, defaults), and which attributes exist.

Using a manifest is optional but recommended to catch errors early. The `Context` class aggregates manifests from all registered providers and validates requests before forwarding to a provider.

For details on implementing a manifest, see [Context & Providers](./06-context.md#manifests).

---

## Examples in Practice

```text
# Simple close above 200-day SMA
close > sma(period=200)

# MACD crossover (line above signal)
macd(fast=12, slow=26).line > macd(fast=12, slow=26).signal

# RSI oversold with rising price
rsi(period=14).value < 30 and rising(close, 3)

# Price breakout above previous high with volume spike
close > high[1] and volume > volume[1] * 1.5
```

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
- **[Contributing](./10-contributing.md)** – development setup and guidelines