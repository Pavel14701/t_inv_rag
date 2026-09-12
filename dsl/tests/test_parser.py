"""Unit tests for the DSL parser.

This module tests the Parser class, which converts a token stream into an AST.
It covers all grammar rules: literals, indicators, arithmetic, comparisons,
logical operators, let-bindings, historical access, rising/falling functions,
and error handling.
"""

import pytest

from ..ast import (
    Add,
    ASTNode,
    Comparison,
    Div,
    Falling,
    HistoricalAccess,
    IndicatorAccess,
    IndicatorWithParams,
    Let,
    LogicalBinOp,
    LogicalNot,
    Mod,
    Mul,
    MultiComparison,
    Number,
    Pow,
    Rising,
    Sub,
    UnaryMinus,
    Var,
)
from ..exceptions import ParseError
from ..parser import Parser


def parse(code: str) -> ASTNode:
    """Helper to parse a DSL expression and return the root AST node."""
    p = Parser()
    return p.parse(code)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_simple_number() -> None:
    """Test parsing of a numeric literal."""
    ast = parse("42")
    assert isinstance(ast, Number)
    assert ast.value == 42.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_negative_number() -> None:
    """Test parsing of a negative number via unary minus."""
    ast = parse("-42")
    assert isinstance(ast, UnaryMinus)
    assert isinstance(ast.operand, Number)
    assert ast.operand.value == 42.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_access() -> None:
    """Test parsing of a simple indicator without attributes."""
    ast = parse("close")
    assert isinstance(ast, IndicatorAccess)
    assert ast.indicator == "close"
    assert ast.attributes == []


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_with_attribute() -> None:
    """Test parsing of an indicator with a dot attribute."""
    ast = parse("rsi.value")
    assert isinstance(ast, IndicatorAccess)
    assert ast.indicator == "rsi"
    assert ast.attributes == ["value"]


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_with_multiple_attributes() -> None:
    """Test parsing of an indicator with multiple dot-separated attributes."""
    ast = parse("rsi.value.signal")
    assert isinstance(ast, IndicatorAccess)
    assert ast.indicator == "rsi"
    assert ast.attributes == ["value", "signal"]


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_with_params() -> None:
    """Test parsing of an indicator with named parameters."""
    ast = parse("rsi(period=14).value")
    assert isinstance(ast, IndicatorWithParams)
    assert ast.indicator == "rsi"
    assert "period" in ast.params
    assert ast.attributes == ["value"]
    # Ensure parameter expression is Number
    param = ast.params["period"]
    assert isinstance(param, Number)
    assert param.value == 14.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_with_multiple_params() -> None:
    """Test parsing of an indicator with multiple parameters."""
    ast = parse("macd(fast=12, slow=26).line")
    assert isinstance(ast, IndicatorWithParams)
    assert ast.indicator == "macd"
    assert "fast" in ast.params and "slow" in ast.params
    assert ast.attributes == ["line"]


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_indicator_with_params_and_attributes_and_history() -> None:
    """Test parsing of an indicator with parameters, attributes,
    and historical offset.
    """
    ast = parse("rsi(period=14).value[1]")
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 1
    inner = ast.expr
    assert isinstance(inner, IndicatorWithParams)
    assert inner.indicator == "rsi"
    assert inner.attributes == ["value"]


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_add() -> None:
    """Test parsing of addition."""
    ast = parse("1 + 2")
    assert isinstance(ast, Add)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 1.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 2.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_sub() -> None:
    """Test parsing of subtraction."""
    ast = parse("5 - 3")
    assert isinstance(ast, Sub)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 5.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_mul() -> None:
    """Test parsing of multiplication."""
    ast = parse("2 * 3")
    assert isinstance(ast, Mul)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 2.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_div() -> None:
    """Test parsing of division."""
    ast = parse("10 / 2")
    assert isinstance(ast, Div)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 10.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 2.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_mod() -> None:
    """Test parsing of modulo."""
    ast = parse("10 % 3")
    assert isinstance(ast, Mod)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 10.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_arithmetic_pow() -> None:
    """Test parsing of exponentiation (right-associative)."""
    ast = parse("2 ^ 3")
    assert isinstance(ast, Pow)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 2.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_precedence_mul_add() -> None:
    """Test that multiplication has higher precedence than addition."""
    ast = parse("1 + 2 * 3")
    assert isinstance(ast, Add)
    assert isinstance(ast.right, Mul)
    assert isinstance(ast.right.left, Number)
    assert ast.right.left.value == 2.0
    assert isinstance(ast.right.right, Number)
    assert ast.right.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_precedence_pow_mul() -> None:
    """Test that exponentiation has higher precedence than multiplication."""
    ast = parse("2 * 3 ^ 2")
    assert isinstance(ast, Mul)
    assert isinstance(ast.right, Pow)
    assert isinstance(ast.right.left, Number)
    assert ast.right.left.value == 3.0
    assert isinstance(ast.right.right, Number)
    assert ast.right.right.value == 2.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_right_associativity_pow() -> None:
    """Test that exponentiation is right-associative (2^3^2 = 2^(3^2))."""
    ast = parse("2 ^ 3 ^ 2")
    assert isinstance(ast, Pow)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 2.0
    assert isinstance(ast.right, Pow)
    assert isinstance(ast.right.left, Number)
    assert ast.right.left.value == 3.0
    assert isinstance(ast.right.right, Number)
    assert ast.right.right.value == 2.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_unary_minus_precedence() -> None:
    """Test that unary minus has higher precedence than exponentiation."""
    ast = parse("-2 ^ 3")
    # Should be (-2)^3? Actually unary minus is in factor,
    # so it applies before pow
    assert isinstance(ast, Pow)
    assert isinstance(ast.left, UnaryMinus)
    assert isinstance(ast.left.operand, Number)
    assert ast.left.operand.value == 2.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_parentheses_override_precedence() -> None:
    """Test that parentheses override the usual precedence."""
    ast = parse("(1 + 2) * 3")
    assert isinstance(ast, Mul)
    assert isinstance(ast.left, Add)
    assert isinstance(ast.left.left, Number)
    assert ast.left.left.value == 1.0
    assert isinstance(ast.left.right, Number)
    assert ast.left.right.value == 2.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_comparison_single() -> None:
    """Test parsing of a single comparison."""
    ast = parse("close > 100")
    assert isinstance(ast, Comparison)
    assert ast.operator == ">"
    assert isinstance(ast.left, IndicatorAccess)
    assert isinstance(ast.right, Number)
    assert ast.right.value == 100.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_chained_comparison() -> None:
    """Test parsing of a chained comparison (a < b <= c)."""
    ast = parse("1 < x <= 10")
    assert isinstance(ast, MultiComparison)
    assert ast.operators == ["<", "<="]
    assert len(ast.operands) == 3
    assert isinstance(ast.operands[0], Number)
    assert ast.operands[0].value == 1.0
    assert isinstance(ast.operands[1], IndicatorAccess)
    assert ast.operands[1].indicator == "x"
    assert isinstance(ast.operands[2], Number)
    assert ast.operands[2].value == 10.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_chained_comparison_with_different_ops() -> None:
    """Test chained comparison with mixed operators (e.g., a < b == c)."""
    ast = parse("a < b == c")
    assert isinstance(ast, MultiComparison)
    assert ast.operators == ["<", "=="]
    assert len(ast.operands) == 3


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_logical_and_or() -> None:
    """Test that 'and' and 'or' are parsed with
    correct precedence (and higher than or).
    """
    ast = parse("a > 0 and b < 10 or c == 5")
    assert isinstance(ast, LogicalBinOp)
    assert ast.operator == "or"
    # left operand is an 'and'
    assert isinstance(ast.left, LogicalBinOp)
    assert ast.left.operator == "and"
    # right operand is comparison
    assert isinstance(ast.right, Comparison)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_logical_not() -> None:
    """Test parsing of logical NOT."""
    ast = parse("not (a > b)")
    assert isinstance(ast, LogicalNot)
    assert isinstance(ast.operand, Comparison)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_not_precedence() -> None:
    """Test that NOT has higher precedence than AND/OR."""
    ast = parse("not a and b")
    # Should be (not a) and b
    assert isinstance(ast, LogicalBinOp)
    assert ast.operator == "and"
    assert isinstance(ast.left, LogicalNot)
    # ast.left.operand will be IndicatorAccess
    assert isinstance(ast.left.operand, IndicatorAccess)
    assert ast.left.operand.indicator == "a"


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_complex_logical_expression() -> None:
    """Test parsing of a complex logical expression with parentheses."""
    ast = parse("(a > 0 and b < 10) or (c == 5 and d != 0)")
    assert isinstance(ast, LogicalBinOp)
    assert ast.operator == "or"
    assert isinstance(ast.left, LogicalBinOp) and ast.left.operator == "and"
    assert isinstance(ast.right, LogicalBinOp) and ast.right.operator == "and"


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_let_expression() -> None:
    """Test parsing of a let expression."""
    ast = parse("let x = rsi(period=14) in x > 70")
    assert isinstance(ast, Let)
    assert ast.var == "x"
    assert isinstance(ast.value, IndicatorWithParams)
    assert isinstance(ast.body, Comparison)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_let_with_arithmetic() -> None:
    """Test let binding with arithmetic expression."""
    ast = parse("let x = 5 + 3 in x * 2")
    assert isinstance(ast, Let)
    assert isinstance(ast.value, Add)
    assert isinstance(ast.value.left, Number)
    assert ast.value.left.value == 5.0
    assert isinstance(ast.value.right, Number)
    assert ast.value.right.value == 3.0
    assert isinstance(ast.body, Mul)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_nested_let() -> None:
    """Test nested let expressions (inner let shadows outer)."""
    ast = parse("let x = 5 in let y = x + 1 in y > 10")
    assert isinstance(ast, Let)  # outer
    inner_let = ast.body
    assert isinstance(inner_let, Let)
    assert inner_let.var == "y"
    add_node = inner_let.value
    assert isinstance(add_node, Add)
    assert isinstance(add_node.left, Var)
    assert add_node.left.name == "x"
    assert isinstance(add_node.right, Number)
    assert add_node.right.value == 1.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_let_variable_usage() -> None:
    """Test that variables in body are represented as Var nodes."""
    ast = parse("let x = 10 in x + 5")
    assert isinstance(ast, Let)
    body = ast.body
    assert isinstance(body, Add)
    assert isinstance(body.left, Var)
    assert body.left.name == "x"
    assert isinstance(body.right, Number)
    assert body.right.value == 5.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_historical_access() -> None:
    """Test parsing of historical access with offset."""
    ast = parse("close[1]")
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 1
    assert isinstance(ast.expr, IndicatorAccess)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_historical_access_with_attrs() -> None:
    """Test parsing of historical access on an indicator with attributes."""
    ast = parse("rsi.value[2]")
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 2
    assert isinstance(ast.expr, IndicatorAccess)
    assert ast.expr.indicator == "rsi"
    assert ast.expr.attributes == ["value"]


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_historical_access_with_params() -> None:
    """Test parsing of historical access on an indicator with parameters."""
    ast = parse("rsi(period=14).value[3]")
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 3
    assert isinstance(ast.expr, IndicatorWithParams)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_rising_function() -> None:
    """Test parsing of the rising function."""
    ast = parse("rising(close, 5)")
    assert isinstance(ast, Rising)
    assert ast.n == 5
    assert isinstance(ast.expr, IndicatorAccess)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_falling_function() -> None:
    """Test parsing of the falling function."""
    ast = parse("falling(close, 3)")
    assert isinstance(ast, Falling)
    assert ast.n == 3
    assert isinstance(ast.expr, IndicatorAccess)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_rising_with_params() -> None:
    """Test rising with an indicator that has parameters."""
    ast = parse("rising(rsi(period=14).value, 3)")
    assert isinstance(ast, Rising)
    assert ast.n == 3
    assert isinstance(ast.expr, IndicatorWithParams)


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_eof() -> None:
    """Test that parser raises ParseError on unexpected end of input."""
    with pytest.raises(ParseError, match="Unexpected EOF"):
        parse("1 +")


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_unexpected_token() -> None:
    """Test that parser raises ParseError on a misplaced token."""
    with pytest.raises(ParseError, match="Unexpected token"):
        parse(")")


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_missing_rparen() -> None:
    """Test that parser catches missing closing parenthesis."""
    with pytest.raises(ParseError, match="Expected RPAREN"):
        parse("(1 + 2")


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_missing_rbracket() -> None:
    """Test that parser catches missing closing bracket."""
    with pytest.raises(ParseError, match="Expected RBRACKET"):
        parse("close[1")


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_missing_comma() -> None:
    """Test that parser catches missing comma in function call."""
    with pytest.raises(ParseError, match="Expected COMMA"):
        parse("rising(close 5)")  # missing comma


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_bad_attribute_dot() -> None:
    """Test that parser catches dot without following identifier."""
    with pytest.raises(ParseError, match="Expected IDENT"):
        parse("rsi.")


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.error
@pytest.mark.deprecated
def test_parse_error_historical_on_non_indicator() -> None:
    """Test that historical access on arbitrary expression
    is not allowed (should be caught by grammar).
    """
    with pytest.raises(ParseError):
        parse("(close + 1)[1]")  # parser should reject this


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_very_long_expression() -> None:
    """Test parsing a very long expression
    (should not hit recursion limits).
    """
    long_expr = " + ".join(["1"] * 100)
    parse(long_expr)  # should not raise


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_mixed_operators_without_spaces() -> None:
    """Test parsing of expression without spaces (e.g., 1+2)."""
    ast = parse("1+2")
    assert isinstance(ast, Add)
    assert isinstance(ast.left, Number)
    assert ast.left.value == 1.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 2.0


@pytest.mark.unit
@pytest.mark.parser
@pytest.mark.deprecated
def test_multiple_attributes() -> None:
    """Test indicator with three attributes."""
    ast = parse("a.b.c.d")
    assert isinstance(ast, IndicatorAccess)
    assert ast.attributes == ["b", "c", "d"]
