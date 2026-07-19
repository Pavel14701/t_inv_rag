import pytest
from dsl.parser import Parser
from dsl.ast import *
from dsl.exceptions import ParseError

def parse(code):
    p = Parser()
    return p.parse(code)

def test_simple_number():
    ast = parse("42")
    assert isinstance(ast, Number)
    assert ast.value == 42.0

def test_indicator_access():
    ast = parse("close")
    assert isinstance(ast, IndicatorAccess)
    assert ast.indicator == "close"
    assert ast.attributes == []

def test_indicator_with_params():
    ast = parse("rsi(period=14).value")
    assert isinstance(ast, IndicatorWithParams)
    assert ast.indicator == "rsi"
    assert "period" in ast.params
    assert ast.attributes == ["value"]

def test_arithmetic_add():
    ast = parse("1 + 2")
    assert isinstance(ast, Add)
    assert isinstance(ast.left, Number)
    assert isinstance(ast.right, Number)

def test_precedence_mul_add():
    ast = parse("1 + 2 * 3")
    assert isinstance(ast, Add)
    assert isinstance(ast.right, Mul)

def test_comparison_single():
    ast = parse("close > 100")
    assert isinstance(ast, Comparison)
    assert ast.operator == '>'

def test_chained_comparison():
    ast = parse("1 < x <= 10")
    assert isinstance(ast, MultiComparison)
    assert ast.operators == ['<', '<=']
    assert len(ast.operands) == 3

def test_logical_and_or():
    ast = parse("a > 0 and b < 10 or c == 5")
    assert isinstance(ast, LogicalBinOp)
    assert ast.operator == 'or'

def test_not():
    ast = parse("not (a > b)")
    assert isinstance(ast, LogicalNot)

def test_let_expression():
    ast = parse("let x = rsi(period=14) in x > 70")
    assert isinstance(ast, Let)
    assert ast.var == 'x'
    assert isinstance(ast.value, IndicatorWithParams)
    assert isinstance(ast.body, Comparison)

def test_historical_access():
    ast = parse("close[1]")
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 1
    assert isinstance(ast.expr, IndicatorAccess)

def test_rising_function():
    ast = parse("rising(close, 5)")
    assert isinstance(ast, Rising)
    assert ast.n == 5

def test_falling_function():
    ast = parse("falling(close, 3)")
    assert isinstance(ast, Falling)

def test_power():
    ast = parse("2 ^ 3 ^ 2")
    assert isinstance(ast, Pow)
    assert isinstance(ast.left, Number)
    assert isinstance(ast.right, Pow)

def test_unary_minus():
    ast = parse("-x")
    assert isinstance(ast, UnaryMinus)

def test_parentheses():
    ast = parse("(1 + 2) * 3")
    assert isinstance(ast, Mul)
    assert isinstance(ast.left, Add)

def test_parse_error_eof():
    with pytest.raises(ParseError):
        parse("1 +")

def test_parse_error_unexpected():
    with pytest.raises(ParseError):
        parse(")")

def test_let_with_variable_usage():
    ast = parse("let x = 10 in x + 5")
    assert isinstance(ast, Let)
    assert ast.var == 'x'
    body = ast.body
    assert isinstance(body, Add)
    # Теперь левый операнд — Var
    assert isinstance(body.left, Var)
    assert body.left.name == 'x'

def test_let_nested_variable():
    ast = parse("let x = 5 in let y = x + 1 in y > 10")
    assert isinstance(ast, Let)  # внешний let
    inner_let = ast.body
    assert isinstance(inner_let, Let)
    assert inner_let.var == 'y'
    # внутри y = x + 1 : x должно быть Var
    add_node = inner_let.value
    assert isinstance(add_node, Add)
    assert isinstance(add_node.left, Var)
    assert add_node.left.name == 'x'