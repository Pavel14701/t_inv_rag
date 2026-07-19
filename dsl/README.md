# DSL for Technical Indicators in Python

An embeddable domain-specific language (DSL) for describing conditions based on technical indicators. It lets you write complex trading rules in a single line, combining arithmetic, comparisons, logical operators, indicator calls with parameters, historical offsets, `rising`/`falling` functions, and `let` bindings.

**Example:**  
`let r = rsi(period=14) in r > 70 and close[1] < close`  
(RSI above 70 and previous close below the current close)

## Quick Start

1. **Installation** (using `uv`):

```bash
uv init my_project
cd my_project
uv add "git+https://github.com/your/repo.git"
```

2. **First run**:

```python
from dsl.parser import Parser
from dsl.interpreter import Interpreter
from dsl.context import Context

class MyContext(Context):
    def get_value(self, indicator, params, attributes, offset):
        # Logic to retrieve an indicator value
        return 150.0

    def get_history(self, indicator, params, attributes, n):
        # Logic to retrieve historical values
        return [100, 120, 150]

code = "close > 100"
parser = Parser()
ast = parser.parse(code)
ctx = MyContext()
interp = Interpreter(ctx)
result = interp.visit(ast)   # True if the condition holds
print(result)
```

3. **Command-line check** (with `uv run`):

```bash
echo "close > 100" | uv run -m dsl.evaluate
```

## Language Syntax

### Literals and Identifiers

- **Numbers**: `42`, `3.14`, `0.001`
- **Identifiers**: names of indicators or let‑bound variables (start with a letter or `_`, followed by letters/digits/`_`).  
  **Example:** `close`, `rsi`, `my_indicator`

### Arithmetic

`+ - * / % ^`  
`^` – right‑associative exponentiation.

### Comparisons

`<`, `>`, `<=`, `>=`, `==`, `!=`  
Chained comparisons are supported: `a < b <= c`

### Logical Operators

`and`, `or`, `not`  
Short‑circuit evaluation. `0` is treated as `False`, any non‑zero number as `True`.

### Operator Precedence (lowest to highest)

1. Logical: `or`, `and`, `not`
2. Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
3. Addition / subtraction: `+`, `-`
4. Multiplication / division / modulus: `*`, `/`, `%`
5. Unary minus: `-`
6. Exponentiation: `^` (right‑associative)

### Indicators
An indicator name can be any word, e.g. `close`, `volume`, `rsi`, `macd`.  
When referencing an indicator you can specify:

- **Named parameters** in parentheses: `rsi(period=14, source=close)`
- **Attributes** using dot notation: `rsi.value`, `macd.histogram`
- **Historical offset** in square brackets: `close[1]` (previous bar), `close[5]` (5 bars ago).  
  You can combine them: `rsi(period=14).value[1]`.

### `rising` and `falling` Functions

Check whether an indicator has been strictly increasing / decreasing over the last `n` bars.

- `rising(close, 5)` – True if `close` increased over the last 5 bars.
- `falling(volume, 3)` – True if `volume` decreased over the last 3 bars.

**Important:** The argument must be an indicator reference (possibly with attributes), but **not an arbitrary expression** and **not a let‑variable**.

### `let` Expressions
Bind a name to the result of an arbitrary expression and make it available in the body.

```

let name = value_expression in body
```

Example: `let mid = (high + low) / 2 in close > mid`  
`let`‑variables are visible only within the body; the scope ends when the `let` expression finishes. Nesting and shadowing are supported.

## Connecting Your Own Indicators (Implementing `Context`)

All data interaction goes through the abstract `Context` class:

```python
from dsl.context import Context

class MyContext(Context):
    def get_value(self, indicator, params, attributes, offset):
        """
        Must return a float – the indicator value.
        indicator – string (e.g. 'rsi')
        params – dictionary {'param_name': value}
        attributes – list of attribute strings (may be empty)
        offset – integer number of bars back (0 = current)
        """
        pass

    def get_history(self, indicator, params, attributes, n):
        """
        Must return a list of the last n indicator values
        (oldest first, newest last).
        Used by rising/falling.
        """
        pass
```

You can implement data retrieval from any external source: database, REST API, WebSocket, etc.

## Implementation Details

### 1. Scope of `let`

Names introduced by `let` exist only inside the body of that `let`. When the body finishes, the variable is forgotten. Nested `let` expressions create a new scope that shadows outer variables.

```python
let x = 5 in (let x = 10 in x) and x == 5  # True (inner x = 10, outer x remains 5)
```

### 2. Variables vs. Indicators

If a name matches a `let`‑variable, it is resolved as a variable. For example, `let close = 100 in close` yields the number 100, not the indicator `close`. This affects `rising`/`falling` – they require an indicator, so passing a `let`‑variable will raise an error.

### 3. `rising`/`falling` Require an Indicator

The argument must be a plain indicator reference, possibly with attributes, but not an arithmetic expression or a `let`‑variable:

```python
rising(close, 3)            # OK
rising(rsi.value, 5)        # OK
rising(close + 1, 3)        # ERROR
let x = close in rising(x, 3)  # ERROR (x is a number, not an indicator)
```

### 4. Historical Offset

Can only be applied to indicators, not to arbitrary expressions:

```python
close[1]            # OK
(close + 1)[1]      # ParseError
```

### 5. Short‑Circuit Evaluation

`a and b` – if `a` is false, `b` is not evaluated. Same for `a or b`.

### 6. Converting Numbers to Boolean

In a boolean context, `0` and `0.0` are `False`; everything else is `True`.

## Common Pitfalls and How to Avoid Them

### ❌ Passing an expression to `rising`/`falling`

```python
rising(close + 1, 3)   # EvaluationError
```

**Solution:** apply `rising`/`falling` directly to an indicator. If you need an offset, use `[ ]` on the indicator:  
`rising(close[1], 3)` (if the provider supports it) or compute the condition separately.

### ❌ Trying to “shift” a parenthesized expression

```python
(close + 1)[1]   # ParseError
```

s**Solution:** move the offset to the indicator: `close[1] + 1`.

### ❌ Using a `let`‑variable inside `rising`/`falling`

```python
let x = close in rising(x, 3)   # EvaluationError
```

**Solution:** use the indicator directly: `rising(close, 3)`. If you need a variable for another purpose, create it separately:  
`let val = close > 100 in rising(close, 3) and val`

### ❌ Shadowing an indicator name with `let`

```python
let close = 100 in close   # close is now the number 100, not the indicator
```

**Solution:** do not name variables after existing indicators. Use descriptive names like `price`, `rsi_val`, etc.

### ❌ Missing spaces around operators

The DSL does not require semicolons; expressions are separated by whitespace and keywords.  
Mistake: `close>100` – the parser sees a single identifier `close>100`.  
**Solution:** always put spaces around operators: `close > 100`.

## Examples

### Simple conditions

```python
close > 100
volume >= 1000000
rsi > 70
```

### Complex logic

```python
not (close < 50) or (volume > 1000000 and rsi > 70)
```

### Comparison with the previous bar

```python
close > close[1]
```

### Using `let` for clarity

```python
let overbought = rsi > 70 in overbought and volume > 1000000
```

### Combining with `rising`

```python
rising(close, 3) and rsi > 50
```

## Dependencies

- Python 3.10+
- Standard library (re, dataclasses, abc, typing)

No external packages in the DSL core. Tests use `pytest`.

## Running the Tests

```bash
uv run pytest dsl/tests/
```

**Connecting to the real world** – implement the `Context` class and the DSL is ready to use in your trading bot, screener, or analytics dashboard.
