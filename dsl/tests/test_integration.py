import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from dsl.tokenizer import Tokenizer
from dsl.parser import Parser
from dsl.interpreter import Interpreter
from dsl.context import Context
from dsl.exceptions import ParseError, EvaluationError


class TestContext(Context):
    """Контекст с настраиваемыми значениями индикаторов и историей."""

    __test__ = False  # prevent pytest from collecting this class

    def __init__(self, values=None, history=None):
        self.values = values or {}
        self.history = history or {}

    def get_value(self, indicator, params, attributes, offset):
        params_key = tuple(sorted(params.items())) if params else ()
        attrs_key = tuple(attributes) if attributes else ()
        key = (indicator, params_key, attrs_key, offset)
        if key in self.values:
            return self.values[key]
        raise ValueError(f"Value not found for {key}")

    def get_history(self, indicator, params, attributes, n):
        params_key = tuple(sorted(params.items())) if params else ()
        attrs_key = tuple(attributes) if attributes else ()
        key = (indicator, params_key, attrs_key)
        if key in self.history:
            hist = self.history[key]
            return hist[-n:] if len(hist) >= n else hist
        return []


def evaluate(code: str, ctx: Context) -> bool:
    """Tokenise, parse and evaluate a DSL expression."""
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(code)
    parser = Parser()
    parser.tokens = tokens
    parser.pos = 0
    ast = parser._expression()
    if parser.pos < len(tokens):
        raise ParseError(f"Unexpected token at end: {tokens[parser.pos].value}")
    interpreter = Interpreter(ctx)
    return interpreter.visit(ast)


# ---------- Tests ----------

def test_simple_arithmetic_with_indicator():
    ctx = TestContext(values={
        ("close", (), (), 0): 100.0,
        ("volume", (), (), 0): 1000.0,
    })
    assert evaluate("close * 2 + volume / 10 - 50", ctx) == True
    assert evaluate("close - 100", ctx) == False


def test_chained_comparison_with_indicators():
    ctx = TestContext(values={
        ("low", (), (), 0): 50,
        ("close", (), (), 0): 55,
        ("high", (), (), 0): 60,
    })
    assert evaluate("low < close <= high", ctx) == True
    assert evaluate("low > close", ctx) == False


def test_logical_combination():
    ctx = TestContext(values={
        ("rsi", (("period", 14),), ("value",), 0): 80,
        ("close", (), (), 0): 150,
    })
    code = "rsi(period=14).value > 70 and close > 100"
    assert evaluate(code, ctx) == True
    code2 = "not (rsi(period=14).value < 30) or close < 200"
    assert evaluate(code2, ctx) == True


def test_let_with_reuse():
    ctx = TestContext(values={("close", (), (), 0): 200})
    code = "let x = close - 100 in x > 0 and x < 200"
    assert evaluate(code, ctx) == True


def test_nested_let():
    """Nested let expressions with variable shadowing."""
    ctx = TestContext(values={("close", (), (), 0): 10})
    code = "let x = close in let y = x + 1 in y > 10"
    assert evaluate(code, ctx) == True

    code2 = "let x = 5 in (let x = 10 in x) and x == 5"
    assert evaluate(code2, ctx) == True


def test_historical_access_inside_let():
    ctx = TestContext(
        values={("close", (), (), 1): 95},
        history={("close", (), ()): [90, 95]}
    )
    code = "let prev = close[1] in prev < 100"
    assert evaluate(code, ctx) == True


def test_rising_falling_with_let():
    """Rising/falling applied directly to indicators,
    with let used for numeric results."""
    ctx = TestContext(
        values={
            ("close", (), (), 0): 10,
            ("volume", (), (), 0): 100
        },
        history={
            ("close", (), ()): [10, 20, 30],
            ("volume", (), ()): [100, 90, 80],
        }
    )
    # Use let to store a derived value, and apply rising/falling to the indicator
    assert evaluate("let x = close + 1 in rising(close, 2) and x > 10", ctx) == True
    assert evaluate("let y = volume - 10 in falling(volume, 2) and y > 80", ctx) == True


def test_complex_expression_with_operators():
    ctx = TestContext(values={("close", (), (), 0): 10})
    code = "(-close ^ 2) + 5 * close - 3"
    assert evaluate(code, ctx) == True
    code2 = "close > 0 and not (close < 5)"
    assert evaluate(code2, ctx) == True


def test_indicator_with_parameters_and_attributes():
    ctx = TestContext(values={
        ("macd", (("fast",12),("slow",26)), ("line",), 0): 1.5,
        ("macd", (("fast",12),("slow",26)), ("signal",), 0): 0.5,
    })
    code = "macd(fast=12, slow=26).line > macd(fast=12, slow=26).signal"
    assert evaluate(code, ctx) == True


def test_errors():
    """Integration error scenarios."""
    ctx = TestContext(values={("close", (), (), 0): 100})

    # Division by zero
    with pytest.raises(EvaluationError, match="Division by zero"):
        evaluate("1 / 0", ctx)

    # Undefined identifier (treated as unknown indicator)
    with pytest.raises(EvaluationError, match="Indicator error"):
        evaluate("x", ctx)

    # Syntax error (missing operand)
    with pytest.raises(ParseError):
        evaluate("close +", ctx)

    # Historical access is not allowed on arbitrary expressions
    with pytest.raises(ParseError):
        evaluate("(close + 1)[1]", ctx)


def test_empty_input():
    with pytest.raises(ParseError):
        evaluate("", TestContext())


def test_boolean_result_from_number():
    assert evaluate("0", TestContext()) == False
    assert evaluate("0.0", TestContext()) == False
    assert evaluate("1", TestContext()) == True
    assert evaluate("-0.001", TestContext()) == True