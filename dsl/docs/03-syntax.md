# Language Syntax Reference

This document describes the complete grammar of the DSL, including literals, operators, precedence, and special constructs.

---

## Grammar Overview

The language is expression‑based. Every DSL string is a single expression that evaluates to a boolean value.

### EBNF (Simplified)

```text
expression  = let_expr | or_expr
let_expr    = 'let' IDENT '=' or_expr 'in' expression
or_expr     = and_expr ('or' and_expr)*
and_expr    = not_expr ('and' not_expr)*
not_expr    = 'not' not_expr | comparison
comparison  = arith_expr (comp_op arith_expr)*
arith_expr  = term (('+' | '-') term)*
term        = factor (('*' | '/' | '%') factor)*
factor      = unary ('^' factor)?        # right‑associative
unary       = ('-')? atom
atom        = NUMBER
            | IDENT                       # indicator or let‑variable
            | IDENT '(' param_list ')' ('.' IDENT)* ('[' NUMBER ']')?
            | IDENT ('.' IDENT)* ('[' NUMBER ']')?   # indicator with attrs/offset
            | 'rising' '(' atom ',' NUMBER ')'
            | 'falling' '(' atom ',' NUMBER ')'
            | '(' expression ')'

comp_op     = '<' | '>' | '<=' | '>=' | '==' | '!='
param_list  = IDENT '=' expression (',' IDENT '=' expression)*
```

---

## Literals and Identifiers

### Numbers

- Integers: `42`, `0`, `-5` (unary minus)
- Floats: `3.14`, `0.001`, `-0.5`

### Identifiers

Start with a letter or underscore, followed by letters, digits, or underscores. Examples: `close`, `rsi`, `_temp`, `my_indicator`.

---

## Arithmetic Operators

| Operator | Meaning | Precedence | Associativity |
|:---------|:--------|:-----------|:--------------|
| `+`      | Addition | 4 (with `-`) | left |
| `-`      | Subtraction | 4 | left |
| `*`      | Multiplication | 5 | left |
| `/`      | Division | 5 | left |
| `%`      | Modulo | 5 | left |
| `^`      | Exponentiation | 6 | **right** |
| unary `-`| Negation | 5.5 (higher than `^`? Actually unary minus is in `factor` which has higher precedence than `^`, so `-2^3` = `(-2)^3`) | – |

### Example

```python
2 + 3 * 4      # 14 (3*4=12, +2)
2 ^ 3 ^ 2      # 2^(3^2) = 512 (right‑associative)
-2 ^ 3         # (-2)^3 = -8
```

---

## Comparison Operators

| Operator | Meaning |
|:---------|:--------|
| `<`      | Less than |
| `>`      | Greater than |
| `<=`     | Less than or equal |
| `>=`     | Greater than or equal |
| `==`     | Equal |
| `!=`     | Not equal |

### Chained Comparisons

Comparisons can be chained: `a < b <= c`. This is evaluated as `(a < b) and (b <= c)`. All operands are evaluated only once.

```python
close > 100 and close < 200   # two comparisons
100 < close < 200             # same, but more concise
```

---

## Logical Operators

| Operator | Precedence | Associativity |
|:---------|:-----------|:--------------|
| `not`    | highest (applies to the following expression) | right |
| `and`    | middle | left (short‑circuit) |
| `or`     | lowest | left (short‑circuit) |

### Short‑Circuit Evaluation

- `a and b` – if `a` is false, `b` is not evaluated.
- `a or b` – if `a` is true, `b` is not evaluated.

### Boolean Conversion

In a logical context, numbers are converted to boolean: `0` and `0.0` → `False`; everything else → `True`.

```python
5 and 0      # False  (0 is False)
0 or 42      # True   (42 is True)
not 0        # True
```

---

## Full Precedence Table (from highest to lowest)

1. **Parentheses** `( )`, **historical offset** `[ ]` (applied to indicators only)
2. **Unary minus** `-` (applied to an atom)
3. **Exponentiation** `^` (right‑associative)
4. **Multiplication** `*`, **Division** `/`, **Modulo** `%` (left‑associative)
5. **Addition** `+`, **Subtraction** `-` (left‑associative)
6. **Comparisons** `<`, `>`, `<=`, `>=`, `==`, `!=` (non‑associative; chaining allowed)
7. **Logical NOT** `not`
8. **Logical AND** `and`
9. **Logical OR** `or`

> **Note:** `not` has higher precedence than `and` and `or`, but lower than comparisons. Example: `not close > 100` is parsed as `not (close > 100)`.

---

## Whitespace and Comments

- Whitespace (spaces, newlines, tabs) is ignored except as token separators.
- Comments are **not** supported in the current version.

---

## Reserved Keywords

The following are reserved and cannot be used as indicator names or variables:

```text
let, in, and, or, not, rising, falling
```

---

## Grammar Corner Cases

- The parser is recursive descent and uses a hand‑written lexer. It supports:
  - Right‑associativity for `^`.
  - Left‑associativity for arithmetic and logical operators.
  - Chained comparisons (multiple operators with three or more operands).
  - Nested `let` expressions and shadowing.

For more details on the implementation, see [Advanced Topics](./08-advanced.md#parser-implementation).

---

## Examples of Valid and Invalid Syntax

### Valid

```text
close > 100
rsi(period=14).value < 30
close[1] - close > 5
let x = (high + low) / 2 in close > x
rising(close, 3) and volume > 1000000
(close > 100) or (rsi < 30)
```

### Invalid

```text
close>100                 # missing space around >
close +                   # missing right operand
rsi(period=14 value)      # missing equals sign
let x = 5 in x            # missing 'in'? Actually valid: let x=5 in x
rising(close + 1, 3)      # rising expects an indicator expression, not an arithmetic expression
(close + 1)[1]            # historical offset cannot be applied to an arbitrary expression
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
