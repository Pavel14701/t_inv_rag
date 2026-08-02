# Let Expressions – Variable Binding and Scoping

`let` expressions allow you to bind a name to the result of an expression and reuse it inside a body. This improves readability and avoids recomputation.

---

## Syntax

```text
let name = value_expr in body_expr
```

- `name` – an identifier (starts with letter or `_`, followed by letters/digits/`_`).
- `value_expr` – any expression (the value is computed and bound to `name`).
- `body_expr` – any expression (the variable `name` is in scope here).

The entire `let` expression evaluates to the value of `body_expr`.

---

## Scoping Rules

- The variable is **only visible inside the body**. It is **not** visible outside the `let` expression.
- **Nested `let`** – each `let` creates a new scope. Inner scopes can shadow outer variables.
- The scope ends when the body finishes evaluating.

### Example: Shadowing

```python
let x = 5 in (let x = 10 in x) and x == 5
# inner x = 10, outer x remains 5 → True
```

### Example: Nested

```python
let a = 10 in
  let b = a + 5 in
    b > 15
# a = 10, b = 15 → True
```

---

## Variables vs. Indicators

When the parser sees an identifier, it first checks if it matches a variable in the current scope (innermost to outermost). If found, it's treated as a `Var` node. Otherwise, it's an indicator reference (`IndicatorAccess` or `IndicatorWithParams`).

### Example

```python
let close = 100 in close   # close is the number 100, not the indicator
```

**This shadows the indicator `close`.** If you need both, use different names:

```python
let price = close in price > 100   # price = value of indicator close
```

---

## Using `let` to Avoid Repetition

Instead of writing the same sub‑expression multiple times:

```python
# Without let (repeats rsi(14).value)
rsi(14).value > 70 and rsi(14).value < 80

# With let (compute once)
let r = rsi(14).value in r > 70 and r < 80
```

This also makes the code **more readable** and **easier to maintain**.

---

## Using `let` with `rising`/`falling`

`rising` and `falling` require an indicator expression, not a variable. So you **cannot** do this:

```python
let x = close in rising(x, 3)   # ❌ Error: x is a number, not an indicator
```

Instead, use the indicator directly:

```python
rising(close, 3) and close > 100
```

Or if you need to combine with a variable for another purpose:

```python
let val = close > 100 in rising(close, 3) and val
```

---

## Performance Considerations

- **Reuse expensive calculations** – if an indicator is used multiple times, assign it to a `let` variable.
- **Short‑circuit** – if the value is only needed in one branch, you can use `and`/`or` directly instead of `let`. But `let` is clearer when the expression is complex.

---

## Common Patterns

### 1. Decompose complex conditions

```python
let ma = sma(20) in
  let trend = close > ma in
    trend and volume > volume[1]
```

### 2. Use as a temporary variable

```python
let diff = close - close[1] in diff > 0 and diff < 5
```

### 3. Combine with logical operators

```python
let cond = rsi > 70 and volume > 1000000 in cond or (close > high[1])
```

---

## Summary

| Feature | Description |
|:--------|:------------|
| Scope | Body of the `let` expression |
| Shadowing | Inner variables can shadow outer ones |
| Visibility | Not visible outside the body |
| Variable vs Indicator | Variables take precedence over indicator names |
| Performance | Avoids recomputation |

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
