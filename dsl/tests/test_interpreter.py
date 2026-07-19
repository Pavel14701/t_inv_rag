import pytest
from dsl.interpreter import Interpreter
from dsl.ast import *
from dsl.context import Context  # базовый класс или интерфейс
from dsl.exceptions import EvaluationError

class FakeContext(Context):
    def __init__(self, values=None, history=None):
        self.values = values or {}
        self.history = history or {}

    def get_value(self, indicator, params, attributes, offset):
        key = (indicator, tuple(sorted(params.items())), tuple(attributes), offset)
        if key in self.values:
            return self.values[key]
        raise ValueError(f"No value for {key}")

    def get_history(self, indicator, params, attributes, n):
        key = (indicator, tuple(sorted(params.items())), tuple(attributes))
        if key in self.history:
            hist = self.history[key]
            return hist[-n:] if len(hist) >= n else hist
        return []

def parse(code):
    from dsl.parser import Parser
    p = Parser()
    return p.parse(code)

def test_number():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    ast = parse("42")
    assert interp.visit(ast) == True   # 42 != 0
    ast0 = parse("0")
    assert interp.visit(ast0) == False

def test_arithmetic():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    assert interp.visit(parse("1 + 2")) == True
    assert interp.visit(parse("5 - 5")) == False
    assert interp.visit(parse("2 * 0")) == False
    with pytest.raises(EvaluationError, match="Division by zero"):
        interp.visit(parse("1 / 0"))

def test_indicator_access():
    ctx = FakeContext({
        ("close", (), (), 0): 150.0
    })
    interp = Interpreter(ctx)
    assert interp.visit(parse("close")) == True
    assert interp.visit(parse("close > 100")) == True

def test_comparison():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    assert interp.visit(parse("5 > 3")) == True
    assert interp.visit(parse("5 < 3")) == False
    assert interp.visit(parse("5 == 5")) == True
    assert interp.visit(parse("5 != 5")) == False

def test_logical():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    assert interp.visit(parse("1 and 1")) == True
    assert interp.visit(parse("1 and 0")) == False
    assert interp.visit(parse("0 or 1")) == True
    assert interp.visit(parse("not 1")) == False

def test_let_simple():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    # let x = 5 in x > 3
    assert interp.visit(parse("let x = 5 in x > 3")) == True

def test_let_with_indicator():
    ctx = FakeContext({
        ("rsi", (("period", 14),), (), 0): 65.0
    })
    interp = Interpreter(ctx)
    ast = parse("let r = rsi(period=14) in r > 70")
    assert interp.visit(ast) == False  # 65 > 70 ложно

def test_historical_access():
    ctx = FakeContext({
        ("close", (), (), 2): 100.0
    })
    interp = Interpreter(ctx)
    assert interp.visit(parse("close[2] > 90")) == True

def test_rising():
    ctx = FakeContext(history={
        ("close", (), ()): [10, 20, 30, 40]
    })
    interp = Interpreter(ctx)
    # rising за последние 3 бара: [20,30,40] строго возрастает
    assert interp.visit(parse("rising(close, 3)")) == True
    # за 5 баров – недостаточно истории
    assert interp.visit(parse("rising(close, 5)")) == False

def test_falling():
    ctx = FakeContext(history={
        ("close", (), ()): [40, 30, 20, 10]
    })
    interp = Interpreter(ctx)
    assert interp.visit(parse("falling(close, 3)")) == True

def test_undefined_variable():
    ctx = FakeContext()
    interp = Interpreter(ctx)
    with pytest.raises(EvaluationError, match="Indicator error"):
        interp.visit(parse("x"))  # без let x считается индикатором