# Examples

This document provides a variety of practical examples using the DSL.

---

## Prerequisites

Assume we have a context `ctx` connected to a data source with the following indicators available:

- `close`, `high`, `low`, `volume`
- `sma(period)`
- `rsi(period)`
- `macd(fast, slow)`

All indicators support the `.value` attribute (or similar). For simplicity, we use `.value` where needed.

---

## Basic Conditions

### 1. Price above a threshold

```python
code = "close > 100"
result = evaluate_dsl(code, ctx)  # True if close > 100
```

### 2. Volume spike

```python
code = "volume > 1000000"
```

### 3. Price within a range

```python
code = "low > 50 and high < 100"
```

---

## Using Indicators with Parameters

### 1. RSI oversold

```python
code = "rsi(period=14).value < 30"
```

### 2. MACD bullish crossover

```python
code = "macd(fast=12, slow=26).line > macd(fast=12, slow=26).signal"
```

### 3. SMA above price (trend filter)

```python
code = "close > sma(period=20)"
```

---

## Historical Access

### 1. Price rising compared to previous bar

```python
code = "close > close[1]"
```

### 2. Price above 5‑day high

```python
code = "close > high[1]"   # previous high
# For 5‑day high, you would need a function or combine conditions:
code = "close > high[1] and close > high[2] and close > high[3] and close > high[4]"
# Or use rising:
code = "rising(close, 5)"   # not exactly the same, but similar
```

### 3. Volume spike relative to previous bar

```python
code = "volume > volume[1] * 1.5"
```

---

## Logical Combinations

### 1. Bullish breakout with volume

```python
code = "close > high[1] and volume > volume[1] * 1.2"
```

### 2. Oversold with positive momentum

```python
code = "rsi(14).value < 30 and close > close[1]"
```

### 3. Not in overbought zone

```python
code = "not (rsi(14).value > 80)"
```

---

## Using `let`

### 1. Factor out repeated calculation

```python
code = """
let r = rsi(14).value in
  r < 30 and r > 20
"""
```

### 2. Combine multiple conditions

```python
code = """
let trend = close > sma(20) in
  trend and volume > 1000000
"""
```

### 3. Nested `let`

```python
code = """
let a = rsi(14).value in
  let b = sma(20) in
    a < 30 and close > b
"""
```

---

## Rising and Falling

### 1. Uptrend

```python
code = "rising(close, 5)"
```

### 2. Downtrend with RSI confirmation

```python
code = "falling(close, 3) and rsi(14).value < 50"
```

### 3. Rising volume on breakout

```python
code = "rising(volume, 3) and close > high[1]"
```

---

## Complex Strategies

### 1. Golden Cross (SMA crossover)

```python
code = "sma(period=50) > sma(period=200)"
```

### 2. Bullish MACD crossover with RSI filter

```python
code = """
let macd_line = macd(fast=12, slow=26).line in
let signal = macd(fast=12, slow=26).signal in
  macd_line > signal and rsi(14).value > 50
"""
```

### 3. Breakout with volume confirmation and trailing stop

```python
code = """
let break = close > high[1] * 1.02 in
  break and volume > volume[1] * 1.5 and close > sma(20)
"""
```

### 4. Mean reversion (price below lower Bollinger band?)

Bollinger Bands are not built-in, but you could implement them as a custom indicator. Example with a simplified version:

```python
code = "close < sma(20) - 2 * stddev(20)"
```

---

## Error‑Prone Patterns (and how to fix them)

### ❌ Using an expression in `rising`

```python
# Error: rising expects an indicator
rising(close + 1, 3)
```

✅ **Fix:** Apply `rising` to the indicator directly.

```python
rising(close, 3)
```

### ❌ Using a `let` variable in `rising`

```python
let x = close in rising(x, 3)  # Error: x is a number
```

✅ **Fix:** Use the indicator directly.

```python
rising(close, 3)
```

### ❌ Historical offset on arbitrary expression

```python
(close + 1)[1]   # ParseError
```

✅ **Fix:** Move the offset to the indicator.

```python
close[1] + 1
```

---

## Testing Your Strategy

1. Write your DSL rule.
2. Evaluate it against historical data to see when it triggers.
3. Backtest the triggers.

You can do this by iterating over a time series and applying the DSL at each step, using the context that provides the indicator values for that point in time.

---

## Summary

The DSL is flexible enough to express a wide range of trading conditions. By combining indicators, parameters, attributes, historical offsets, and `let` bindings, you can write clear, concise, and maintainable rules.

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
