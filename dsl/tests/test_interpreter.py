"""Unit tests for the interpreter.

This module tests the Interpreter class directly, using the AST nodes
without going through the full parser. It tests each node type in isolation.
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
from ..exceptions import EvaluationError, ProviderError
from ..interpreter import Interpreter


@pytest.mark.unit
@pytest.mark.interpreter
def test_number_node(context_empty) -> None:
    """Test evaluation of Number node."""
    interp = Interpreter(context_empty)
    assert interp.visit(Number(value=42)) is True
    assert interp.visit(Number(value=0)) is False
    assert interp.visit(Number(value=-0.0)) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_var_node(context_empty) -> None:
    """Test evaluation of Var node (requires let binding)."""
    interp = Interpreter(context_empty)
    node = Let(var="x", value=Number(value=5), body=Var(name="x"))
    # The interpreter sets _locals during Let evaluation
    assert interp.visit(node) is True  # 5 != 0
    # Undefined variable should raise
    with pytest.raises(EvaluationError, match="Undefined variable"):
        interp.visit(Var(name="undefined"))


@pytest.mark.unit
@pytest.mark.interpreter
def test_add_node(context_empty) -> None:
    """Test addition."""
    interp = Interpreter(context_empty)
    node = Add(left=Number(value=5), right=Number(value=3))
    assert interp.visit(node) is True  # 8 != 0
    node = Add(left=Number(value=0), right=Number(value=0))
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_sub_node(context_empty) -> None:
    """Test subtraction."""
    interp = Interpreter(context_empty)
    node = Sub(left=Number(value=5), right=Number(value=3))
    assert interp.visit(node) is True  # 2 != 0
    node = Sub(left=Number(value=5), right=Number(value=5))
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_mul_node(context_empty) -> None:
    """Test multiplication."""
    interp = Interpreter(context_empty)
    node = Mul(left=Number(value=2), right=Number(value=3))
    assert interp.visit(node) is True  # 6 != 0
    node = Mul(left=Number(value=5), right=Number(value=0))
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_div_node(context_empty) -> None:
    """Test division."""
    interp = Interpreter(context_empty)
    node = Div(left=Number(value=10), right=Number(value=2))
    assert interp.visit(node) is True  # 5 != 0
    with pytest.raises(EvaluationError, match="Division by zero"):
        interp.visit(Div(left=Number(value=1), right=Number(value=0)))


@pytest.mark.unit
@pytest.mark.interpreter
def test_mod_node(context_empty) -> None:
    """Test modulo."""
    interp = Interpreter(context_empty)
    node = Mod(left=Number(value=10), right=Number(value=3))
    assert interp.visit(node) is True  # 1 != 0
    with pytest.raises(EvaluationError, match="Modulo by zero"):
        interp.visit(Mod(left=Number(value=1), right=Number(value=0)))


@pytest.mark.unit
@pytest.mark.interpreter
def test_pow_node(context_empty) -> None:
    """Test exponentiation."""
    interp = Interpreter(context_empty)
    node = Pow(left=Number(value=2), right=Number(value=3))
    assert interp.visit(node) is True  # 8 != 0
    node = Pow(left=Number(value=0), right=Number(value=5))
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_unary_minus_node(context_empty) -> None:
    """Test unary minus."""
    interp = Interpreter(context_empty)
    node = UnaryMinus(operand=Number(value=5))
    assert interp.visit(node) is True  # -5 != 0
    node = UnaryMinus(operand=Number(value=0))
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_comparison_node(context_empty) -> None:
    """Test binary comparison."""
    interp = Interpreter(context_empty)
    # Test each operator
    assert (
        interp.visit(
            Comparison(
                operator="<", left=Number(value=5), right=Number(value=10)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator="<", left=Number(value=10), right=Number(value=5)
            )
        )
        is False
    )
    assert (
        interp.visit(
            Comparison(
                operator=">", left=Number(value=10), right=Number(value=5)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator=">", left=Number(value=5), right=Number(value=10)
            )
        )
        is False
    )
    assert (
        interp.visit(
            Comparison(
                operator="<=", left=Number(value=5), right=Number(value=5)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator="<=", left=Number(value=6), right=Number(value=5)
            )
        )
        is False
    )
    assert (
        interp.visit(
            Comparison(
                operator=">=", left=Number(value=5), right=Number(value=5)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator=">=", left=Number(value=4), right=Number(value=5)
            )
        )
        is False
    )
    assert (
        interp.visit(
            Comparison(
                operator="==", left=Number(value=5), right=Number(value=5)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator="==", left=Number(value=5), right=Number(value=6)
            )
        )
        is False
    )
    assert (
        interp.visit(
            Comparison(
                operator="!=", left=Number(value=5), right=Number(value=6)
            )
        )
        is True
    )
    assert (
        interp.visit(
            Comparison(
                operator="!=", left=Number(value=5), right=Number(value=5)
            )
        )
        is False
    )
    with pytest.raises(EvaluationError, match="Unknown comparison operator"):
        interp.visit(
            Comparison(
                operator="???", left=Number(value=1), right=Number(value=2)
            )
        )


@pytest.mark.unit
@pytest.mark.interpreter
def test_multi_comparison_node(context_empty) -> None:
    """Test chained comparison."""
    interp = Interpreter(context_empty)
    # 5 < 10 <= 10
    node = MultiComparison(
        operators=["<", "<="],
        operands=[Number(value=5), Number(value=10), Number(value=10)],
    )
    assert interp.visit(node) is True
    # 5 < 10 > 10
    node = MultiComparison(
        operators=["<", ">"],
        operands=[Number(value=5), Number(value=10), Number(value=10)],
    )
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_logical_binop_node(context_empty) -> None:
    """Test AND and OR."""
    interp = Interpreter(context_empty)
    # AND
    assert (
        interp.visit(
            LogicalBinOp(
                operator="and", left=Number(value=1), right=Number(value=1)
            )
        )
        is True
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="and", left=Number(value=1), right=Number(value=0)
            )
        )
        is False
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="and", left=Number(value=0), right=Number(value=1)
            )
        )
        is False
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="and", left=Number(value=0), right=Number(value=0)
            )
        )
        is False
    )
    # OR
    assert (
        interp.visit(
            LogicalBinOp(
                operator="or", left=Number(value=1), right=Number(value=1)
            )
        )
        is True
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="or", left=Number(value=1), right=Number(value=0)
            )
        )
        is True
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="or", left=Number(value=0), right=Number(value=1)
            )
        )
        is True
    )
    assert (
        interp.visit(
            LogicalBinOp(
                operator="or", left=Number(value=0), right=Number(value=0)
            )
        )
        is False
    )
    # Unknown operator
    with pytest.raises(EvaluationError, match="Unknown logical operator"):
        interp.visit(
            LogicalBinOp(
                operator="xor", left=Number(value=1), right=Number(value=0)
            )
        )


@pytest.mark.unit
@pytest.mark.interpreter
def test_logical_not_node(context_empty) -> None:
    """Test NOT."""
    interp = Interpreter(context_empty)
    assert interp.visit(LogicalNot(operand=Number(value=0))) is True
    assert interp.visit(LogicalNot(operand=Number(value=1))) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_let_node(context_empty) -> None:
    """Test let binding with numeric and boolean values."""
    interp = Interpreter(context_empty)
    # Numeric binding
    node = Let(
        var="x",
        value=Number(value=5),
        body=Comparison(
            operator=">", left=Var(name="x"), right=Number(value=3)
        ),
    )
    assert interp.visit(node) is True
    # Boolean binding
    node = Let(
        var="y", value=Number(value=0), body=LogicalNot(operand=Var(name="y"))
    )
    assert interp.visit(node) is True


@pytest.mark.unit
@pytest.mark.interpreter
def test_historical_access_node(context_with_history) -> None:
    """Test historical access evaluation."""
    interp = Interpreter(context_with_history)
    node = HistoricalAccess(expr=IndicatorAccess(indicator="close"), offset=1)
    assert interp.visit(node) is True
    with pytest.raises(ProviderError):
        interp.visit(
            HistoricalAccess(expr=IndicatorAccess(indicator="close"), offset=2)
        )


@pytest.mark.unit
@pytest.mark.interpreter
def test_rising_node(context_with_sample_data) -> None:
    """Test rising function."""
    interp = Interpreter(context_with_sample_data)
    # close history [10, 20, 30, 40]
    node = Rising(expr=IndicatorAccess(indicator="close"), n=2)
    assert interp.visit(node) is True
    node = Rising(expr=IndicatorAccess(indicator="close"), n=5)
    # insufficient history, should be False
    assert interp.visit(node) is False
    # invalid expression
    with pytest.raises(EvaluationError, match=r"rising.*indicator expression"):
        interp.visit(Rising(expr=Number(value=5), n=2))


@pytest.mark.unit
@pytest.mark.interpreter
def test_falling_node(context_with_sample_data) -> None:
    """Test falling function."""
    # Modify history to decreasing sequence
    provider = context_with_sample_data.providers[0]
    provider.history["close", (), ()] = [40, 30, 20, 10]
    interp = Interpreter(context_with_sample_data)
    node = Falling(expr=IndicatorAccess(indicator="close"), n=2)
    assert interp.visit(node) is True
    node = Falling(expr=IndicatorAccess(indicator="close"), n=5)
    assert interp.visit(node) is False


@pytest.mark.unit
@pytest.mark.interpreter
def test_indicator_with_params_node(context_with_sample_data) -> None:
    """Test indicator access with parameters."""
    interp = Interpreter(context_with_sample_data)
    params: dict[str, ASTNode] = {"period": Number(value=14)}
    node = IndicatorWithParams(
        indicator="rsi", params=params, attributes=["value"]
    )
    assert interp.visit(node) is True  # 80 != 0
    # Unknown indicator should raise
    with pytest.raises(ValueError, match="Unknown indicator: unknown"):
        interp.visit(
            IndicatorWithParams(indicator="unknown", params={}, attributes=[])
        )


@pytest.mark.unit
@pytest.mark.interpreter
def test_division_by_zero_in_subexpression(context_empty) -> None:
    """Test that division by zero in subexpression raises EvaluationError."""
    interp = Interpreter(context_empty)
    node = Add(
        left=Number(value=5),
        right=Div(left=Number(value=1), right=Number(value=0)),
    )
    with pytest.raises(EvaluationError, match="Division by zero"):
        interp.visit(node)


@pytest.mark.unit
@pytest.mark.interpreter
def test_unknown_node_type(context_empty) -> None:
    """Test that visiting an unknown node type raises EvaluationError."""

    class UnknownNode:
        pass

    interp = Interpreter(context_empty)
    with pytest.raises(EvaluationError, match="Unknown AST node"):
        interp.visit(UnknownNode())  # type: ignore


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_number_node(async_context_with_sample_data) -> None:
    """Test asynchronous evaluation of Number node."""
    interp = Interpreter(async_context_with_sample_data)
    assert await interp.visit_async(Number(value=42)) is True
    assert await interp.visit_async(Number(value=0)) is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_arithmetic(async_context_with_sample_data) -> None:
    """Test asynchronous arithmetic evaluation."""
    interp = Interpreter(async_context_with_sample_data)
    node_add = Add(left=Number(value=5), right=Number(value=3))
    assert await interp.visit_async(node_add) is True
    node_div = Div(left=Number(value=10), right=Number(value=2))
    assert await interp.visit_async(node_div) is True
    with pytest.raises(EvaluationError, match="Division by zero"):
        await interp.visit_async(
            Div(left=Number(value=1), right=Number(value=0))
        )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_indicator_access(async_context_with_sample_data) -> None:
    """Test asynchronous indicator access."""
    interp = Interpreter(async_context_with_sample_data)
    node_ind_acc = IndicatorAccess(indicator="close")
    assert await interp.visit_async(node_ind_acc) is True  # 100.0 != 0
    node_comp = Comparison(
        operator=">",
        left=IndicatorAccess(indicator="close"),
        right=Number(value=50),
    )
    assert await interp.visit_async(node_comp) is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_historical_access(async_context_with_sample_data) -> None:
    """Test asynchronous historical access."""
    interp = Interpreter(async_context_with_sample_data)
    node = HistoricalAccess(expr=IndicatorAccess(indicator="close"), offset=1)
    assert await interp.visit_async(node) is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_rising(async_context_with_sample_data) -> None:
    """Test asynchronous rising function."""
    interp = Interpreter(async_context_with_sample_data)
    node = Rising(expr=IndicatorAccess(indicator="close"), n=2)
    assert await interp.visit_async(node) is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.interpreter
@pytest.mark.async_test
async def test_async_falling(async_context_with_sample_data) -> None:
    """Test asynchronous falling function."""
    # Set decreasing history
    provider = async_context_with_sample_data.providers[0]
    provider.history["close", (), ()] = [40, 30, 20, 10]
    interp = Interpreter(async_context_with_sample_data)
    node = Falling(expr=IndicatorAccess(indicator="close"), n=2)
    assert await interp.visit_async(node) is True
