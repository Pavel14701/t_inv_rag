# Indicators, Parameters, Attributes, and Offsets

This document explains how to use technical indicators in the DSL – from simple names to complex calls with parameters, attributes, and historical offsets.

---

## Indicator Name

Any identifier (that is not a reserved keyword) can be an indicator. Examples:

```text
close
volume
rsi
macd
sma
```

The DSL does **not** know which indicators exist in advance – that is up to the `Context` implementation and the manifest (if used).

---

## Named Parameters

Indicators often require parameters (e.g., `period`, `fast`, `slow`). You can pass them by name in parentheses:

```text
rsi(period=14)
macd(fast=12, slow=26)
sma(period=20)
```

Parameter names are identifiers, and their values are arbitrary expressions (not just literals):

```text
rsi(period=14 + 1)           # period = 15
sma(period=close > 100 ? 10 : 20)   # (conditional not supported, but you can use arithmetic/logic)
```

In practice, parameters are usually numeric literals or simple expressions, but the grammar allows any expression.

---

## Attributes

Indicators may return multiple values (e.g., `rsi` returns a numeric value, `macd` may return `line`, `signal`, and `histogram`). Use dot notation:

```text
rsi.value
macd.line
macd.signal
macd.histogram
```

Multiple attributes are allowed: `a.b.c` is parsed as `indicator='a'` with `attributes=['b','c']`. However, typical usage is one attribute.

You can combine parameters and attributes:

```text
rsi(period=14).value
macd(fast=12, slow=26).signal
```

---

## Historical Offset

Access past values by appending `[N]` where `N` is an integer (non‑negative). `[0]` is the current bar, `[1]` is the previous bar, etc.

```text
close[1]        # previous close
rsi(14).value[2]   # RSI value two bars ago
high[0]         # current high (same as high)
```

**Important restrictions:**

- Historical offset can only be applied to an **indicator access** (with or without parameters/attributes). It **cannot** be applied to arbitrary expressions:
  - `(close + 1)[1]` → `ParseError`
  - `(rsi.value)[1]` → `ParseError` (parentheses are not allowed around indicators before offset)
- The offset is a **literal integer** (no expressions allowed).

---

## Rising and Falling Functions

These are special functions that check monotonicity over a window.

```text
rising(expr, n)
falling(expr, n)
```

- `expr` must be an indicator access (with optional parameters and attributes). It **must not** be an arbitrary expression or a let‑bound variable.
- `n` is a literal integer (number of bars to check, including the current bar? Actually the definition: `rising(close, 3)` checks if `close` has increased over the last 3 bars, i.e., `close[0] > close[1] > close[2]`. So it uses the last `n` bars including current.)

### Examples

```text
rising(close, 5)            # True if close increased over the last 5 bars
rising(rsi(14).value, 3)    # True if RSI increased over the last 3 bars
falling(volume, 2)          # True if volume decreased over the last 2 bars
```

### Invalid Usages

```text
rising(close + 1, 3)        # Error: expects an indicator
let x = close in rising(x, 3)  # Error: x is a variable, not an indicator
rising(close[1], 3)         # This is allowed? Yes, because close[1] is an indicator access (with offset). However, the history retrieval will treat it as the base indicator with an offset? Actually rising receives the expression and then uses get_history on the base indicator. If the expression is close[1], it's still an indicator access, but rising will call get_history for the base indicator (close) and then compare historical values. The offset inside the expression is not considered by rising; rising will use the expression as the indicator and retrieve history for that indicator, but the offset is not part of the history request. So rising(close[1], 3) might not behave as expected: it will compare close[1] over the last 3 bars? Actually the DSL implementation: rising evaluates the expression repeatedly? No, the interpreter's _visit_rising_sync extracts the indicator name and params/attrs, and ignores any offset. So it will retrieve history for the indicator without offset, then compare. So rising(close[1], 3) is syntactically allowed but semantically confusing – it will compare close values (not shifted) because the offset is ignored. It's better to avoid that.
```

**Recommendation:** Always use the bare indicator in `rising`/`falling` without an offset. If you need to compare historical values, do the offset in the expression outside:

```text
rising(close, 3) and close[1] < close
```

---

## Indicator Resolution and Manifest

When the interpreter encounters an indicator reference, it calls the context's `get_value` (or `get_value_async`). The context may validate the indicator name, parameters, and attributes against a **manifest**. The manifest describes what indicators are available, which parameters they accept, their types and ranges, and which attributes exist.

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
