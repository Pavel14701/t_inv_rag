"""Unit tests for AST serialization (to_dict / from_dict).

This module tests that all AST nodes can be properly serialized to JSON
and deserialized back, including nested structures and edge cases.
"""

import pytest

from ..ast import (
    Number,
    Var,
    IndicatorAccess,
    IndicatorWithParams,
    Comparison,
    MultiComparison,
    LogicalBinOp,
    LogicalNot,
    Let,
    HistoricalAccess,
    Rising,
    Falling,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    UnaryMinus,
    from_dict,
    ASTNode,
)
from ..parser import Parser


@pytest.mark.unit
@pytest.mark.ast
def test_number_serialization() -> None:
    """Test Number node serialization with positive,
    negative, and zero values.
    """
    for value in (42.5, -3.14, 0.0):
        node = Number(value=value)
        data = node.to_dict()
        assert data == {'type': 'Number', 'value': value}
        reconstructed = Number.from_dict(data)
        assert reconstructed.value == value
        assert reconstructed.type == 'Number'


@pytest.mark.unit
@pytest.mark.ast
def test_var_serialization() -> None:
    """Test Var node serialization with different variable names."""
    for name in ('x', 'my_var', '_temp'):
        node = Var(name=name)
        data = node.to_dict()
        assert data == {'type': 'Var', 'name': name}
        reconstructed = Var.from_dict(data)
        assert reconstructed.name == name


@pytest.mark.unit
@pytest.mark.ast
def test_indicator_access_serialization() -> None:
    """Test IndicatorAccess serialization with and without attributes."""
    # No attributes
    node = IndicatorAccess(indicator='close')
    data = node.to_dict()
    assert data == {
        'type': 'IndicatorAccess',
        'indicator': 'close',
        'attributes': []
    }
    reconstructed = IndicatorAccess.from_dict(data)
    assert reconstructed.indicator == 'close'
    assert reconstructed.attributes == []

    # With attributes
    node = IndicatorAccess(indicator='rsi', attributes=['value', 'signal'])
    data = node.to_dict()
    assert data == {
        'type': 'IndicatorAccess',
        'indicator': 'rsi',
        'attributes': ['value', 'signal']
    }
    reconstructed = IndicatorAccess.from_dict(data)
    assert reconstructed.indicator == 'rsi'
    assert reconstructed.attributes == ['value', 'signal']


@pytest.mark.unit
@pytest.mark.ast
def test_indicator_with_params_serialization() -> None:
    """Test IndicatorWithParams serialization with various parameter types."""
    # Single integer parameter
    params: dict[str, ASTNode] = {'period': Number(value=14.0)}
    node = IndicatorWithParams(
        indicator='rsi',
        params=params,
        attributes=['value']
    )
    data = node.to_dict()
    assert data['type'] == 'IndicatorWithParams'
    assert data['indicator'] == 'rsi'
    assert 'period' in data['params']
    assert data['attributes'] == ['value']
    reconstructed = IndicatorWithParams.from_dict(data)
    assert reconstructed.indicator == 'rsi'
    assert isinstance(reconstructed.params['period'], Number)
    assert reconstructed.params['period'].value == 14.0
    assert reconstructed.attributes == ['value']
    # Multiple parameters, including float
    params_multi: dict[str, ASTNode] = {
        'fast': Number(value=12.0),
        'slow': Number(value=26.5)
    }
    node = IndicatorWithParams(
        indicator='macd',
        params=params_multi,
        attributes=['line', 'signal']
    )
    data = node.to_dict()
    reconstructed = IndicatorWithParams.from_dict(data)
    assert reconstructed.indicator == 'macd'
    assert isinstance(reconstructed.params['fast'], Number)
    assert reconstructed.params['fast'].value == 12.0
    assert isinstance(reconstructed.params['slow'], Number)
    assert reconstructed.params['slow'].value == 26.5
    assert reconstructed.attributes == ['line', 'signal']
    # Empty params
    node = IndicatorWithParams(indicator='close', params={}, attributes=[])
    data = node.to_dict()
    reconstructed = IndicatorWithParams.from_dict(data)
    assert reconstructed.indicator == 'close'
    assert reconstructed.params == {}
    assert reconstructed.attributes == []


@pytest.mark.unit
@pytest.mark.ast
def test_comparison_serialization() -> None:
    """Test Comparison serialization with all operators."""
    operators = ['<', '>', '<=', '>=', '==', '!=']
    for op in operators:
        node = Comparison(
            operator=op,
            left=Number(value=5.0),
            right=Number(value=3.0)
        )
        data = node.to_dict()
        assert data['type'] == 'Comparison'
        assert data['operator'] == op
        reconstructed = Comparison.from_dict(data)
        assert reconstructed.operator == op
        assert isinstance(reconstructed.left, Number)
        assert reconstructed.left.value == 5.0
        assert isinstance(reconstructed.right, Number)
        assert reconstructed.right.value == 3.0


@pytest.mark.unit
@pytest.mark.ast
def test_multi_comparison_serialization() -> None:
    """Test MultiComparison serialization (chained comparisons)."""
    # Chain: 1 < x <= 10
    operands: list[ASTNode] = [
        Number(value=1.0),
        Var(name='x'),
        Number(value=10.0)
    ]
    node = MultiComparison(operators=['<', '<='], operands=operands)
    data = node.to_dict()
    assert data['type'] == 'MultiComparison'
    assert data['operators'] == ['<', '<=']
    assert len(data['operands']) == 3
    reconstructed = MultiComparison.from_dict(data)
    assert reconstructed.operators == ['<', '<=']
    assert len(reconstructed.operands) == 3
    assert isinstance(reconstructed.operands[0], Number)
    assert reconstructed.operands[0].value == 1.0
    assert isinstance(reconstructed.operands[1], Var)
    assert reconstructed.operands[1].name == 'x'
    assert isinstance(reconstructed.operands[2], Number)
    assert reconstructed.operands[2].value == 10.0
    # Chain with 4 operands: a < b == c != d
    operands2: list[ASTNode] = [
        Var(name='a'),
        Var(name='b'),
        Var(name='c'),
        Var(name='d')
    ]
    node = MultiComparison(operators=['<', '==', '!='], operands=operands2)
    data = node.to_dict()
    reconstructed = MultiComparison.from_dict(data)
    assert reconstructed.operators == ['<', '==', '!=']
    assert len(reconstructed.operands) == 4
    assert all(isinstance(op, Var) for op in reconstructed.operands)


@pytest.mark.unit
@pytest.mark.ast
def test_logical_binop_serialization() -> None:
    """Test LogicalBinOp serialization with 'and' and 'or'."""
    for op in ('and', 'or'):
        node = LogicalBinOp(
            operator=op,
            left=Number(value=1.0),
            right=Number(value=0.0)
        )
        data = node.to_dict()
        reconstructed = LogicalBinOp.from_dict(data)
        assert reconstructed.operator == op
        assert isinstance(reconstructed.left, Number)
        assert reconstructed.left.value == 1.0
        assert isinstance(reconstructed.right, Number)
        assert reconstructed.right.value == 0.0


@pytest.mark.unit
@pytest.mark.ast
def test_logical_not_serialization() -> None:
    """Test LogicalNot serialization with nested operand."""
    node = LogicalNot(operand=Number(value=1.0))
    data = node.to_dict()
    reconstructed = LogicalNot.from_dict(data)
    assert isinstance(reconstructed.operand, Number)
    assert reconstructed.operand.value == 1.0

    # Nested not: not (not x)
    node = LogicalNot(operand=LogicalNot(operand=Var(name='x')))
    data = node.to_dict()
    reconstructed = LogicalNot.from_dict(data)
    assert isinstance(reconstructed.operand, LogicalNot)
    assert isinstance(reconstructed.operand.operand, Var)
    assert reconstructed.operand.operand.name == 'x'


@pytest.mark.unit
@pytest.mark.ast
def test_add_serialization() -> None:
    """Test Add serialization."""
    node = Add(left=Number(value=1.0), right=Number(value=2.0))
    data = node.to_dict()
    reconstructed = Add.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 1.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 2.0


@pytest.mark.unit
@pytest.mark.ast
def test_sub_serialization() -> None:
    """Test Sub serialization."""
    node = Sub(left=Number(value=5.0), right=Number(value=3.0))
    data = node.to_dict()
    reconstructed = Sub.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 5.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 3.0


@pytest.mark.unit
@pytest.mark.ast
def test_mul_serialization() -> None:
    """Test Mul serialization."""
    node = Mul(left=Number(value=2.0), right=Number(value=3.0))
    data = node.to_dict()
    reconstructed = Mul.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 2.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 3.0


@pytest.mark.unit
@pytest.mark.ast
def test_div_serialization() -> None:
    """Test Div serialization."""
    node = Div(left=Number(value=10.0), right=Number(value=2.0))
    data = node.to_dict()
    reconstructed = Div.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 10.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 2.0


@pytest.mark.unit
@pytest.mark.ast
def test_mod_serialization() -> None:
    """Test Mod serialization."""
    node = Mod(left=Number(value=10.0), right=Number(value=3.0))
    data = node.to_dict()
    reconstructed = Mod.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 10.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 3.0


@pytest.mark.unit
@pytest.mark.ast
def test_pow_serialization() -> None:
    """Test Pow serialization."""
    node = Pow(left=Number(value=2.0), right=Number(value=3.0))
    data = node.to_dict()
    reconstructed = Pow.from_dict(data)
    assert isinstance(reconstructed.left, Number)
    assert reconstructed.left.value == 2.0
    assert isinstance(reconstructed.right, Number)
    assert reconstructed.right.value == 3.0


@pytest.mark.unit
@pytest.mark.ast
def test_unary_minus_serialization() -> None:
    """Test UnaryMinus serialization with nested operand."""
    node = UnaryMinus(operand=Number(value=5.0))
    data = node.to_dict()
    reconstructed = UnaryMinus.from_dict(data)
    assert isinstance(reconstructed.operand, Number)
    assert reconstructed.operand.value == 5.0

    # Double negation: -(-x)
    node = UnaryMinus(operand=UnaryMinus(operand=Var(name='x')))
    data = node.to_dict()
    reconstructed = UnaryMinus.from_dict(data)
    assert isinstance(reconstructed.operand, UnaryMinus)
    assert isinstance(reconstructed.operand.operand, Var)
    assert reconstructed.operand.operand.name == 'x'


@pytest.mark.unit
@pytest.mark.ast
def test_let_serialization() -> None:
    """Test Let serialization with simple and nested bodies."""
    # Simple let
    node = Let(var='x', value=Number(value=5.0), body=Number(value=10.0))
    data = node.to_dict()
    assert data['type'] == 'Let'
    assert data['var'] == 'x'
    reconstructed = Let.from_dict(data)
    assert reconstructed.var == 'x'
    assert isinstance(reconstructed.value, Number)
    assert reconstructed.value.value == 5.0
    assert isinstance(reconstructed.body, Number)
    assert reconstructed.body.value == 10.0
    # Nested let: let x = 5 in let y = x + 1 in y > 10
    inner_let = Let(
        var='y',
        value=Add(left=Var(name='x'), right=Number(value=1.0)),
        body=Comparison(
            operator='>',
            left=Var(name='y'),
            right=Number(value=10.0)
        )
    )
    outer_let = Let(var='x', value=Number(value=5.0), body=inner_let)
    data = outer_let.to_dict()
    reconstructed = Let.from_dict(data)
    assert reconstructed.var == 'x'
    assert isinstance(reconstructed.value, Number)
    assert reconstructed.value.value == 5.0
    assert isinstance(reconstructed.body, Let)
    assert reconstructed.body.var == 'y'
    assert isinstance(reconstructed.body.value, Add)
    assert isinstance(reconstructed.body.body, Comparison)


@pytest.mark.unit
@pytest.mark.ast
def test_historical_access_serialization() -> None:
    """Test HistoricalAccess serialization with different offsets
    and expression types.
    """
    # Historical access with IndicatorAccess
    for offset in (0, 1, 5):
        node = HistoricalAccess(
            expr=IndicatorAccess(indicator='close'),
            offset=offset
        )
        data = node.to_dict()
        reconstructed = HistoricalAccess.from_dict(data)
        assert reconstructed.offset == offset
        assert isinstance(reconstructed.expr, IndicatorAccess)
        assert reconstructed.expr.indicator == 'close'
    # Historical access with IndicatorWithParams
    indicator_with_params = IndicatorWithParams(
        indicator='rsi',
        params={'period': Number(value=14.0)}
    )
    node_with_params = HistoricalAccess(expr=indicator_with_params, offset=2)
    data = node_with_params.to_dict()
    reconstructed_with_params = HistoricalAccess.from_dict(data)
    assert reconstructed_with_params.offset == 2
    assert isinstance(reconstructed_with_params.expr, IndicatorWithParams)
    assert reconstructed_with_params.expr.indicator == 'rsi'
    assert isinstance(reconstructed_with_params.expr.params['period'], Number)
    assert reconstructed_with_params.expr.params['period'].value == 14.0


@pytest.mark.unit
@pytest.mark.ast
def test_rising_serialization() -> None:
    """Test Rising serialization with different n values."""
    for n in (1, 5, 10):
        node = Rising(expr=IndicatorAccess(indicator='close'), n=n)
        data = node.to_dict()
        reconstructed = Rising.from_dict(data)
        assert reconstructed.n == n
        assert isinstance(reconstructed.expr, IndicatorAccess)
        assert reconstructed.expr.indicator == 'close'


@pytest.mark.unit
@pytest.mark.ast
def test_falling_serialization() -> None:
    """Test Falling serialization with different n values."""
    for n in (1, 5, 10):
        node = Falling(expr=IndicatorAccess(indicator='close'), n=n)
        data = node.to_dict()
        reconstructed = Falling.from_dict(data)
        assert reconstructed.n == n
        assert isinstance(reconstructed.expr, IndicatorAccess)
        assert reconstructed.expr.indicator == 'close'


@pytest.mark.unit
@pytest.mark.ast
def test_from_dict_factory_all_types() -> None:
    """Test the from_dict factory function for all node types
    with type checks.
    """
    # Number
    data = {'type': 'Number', 'value': 42.0}
    node = from_dict(data)
    assert isinstance(node, Number)
    assert node.value == 42.0
    # Var
    data = {'type': 'Var', 'name': 'x'}
    node = from_dict(data)
    assert isinstance(node, Var)
    assert node.name == 'x'
    # IndicatorAccess
    data = {'type': 'IndicatorAccess', 'indicator': 'close', 'attributes': []}
    node = from_dict(data)
    assert isinstance(node, IndicatorAccess)
    assert node.indicator == 'close'
    # IndicatorWithParams
    data = {
        'type': 'IndicatorWithParams',
        'indicator': 'rsi',
        'params': {'period': {'type': 'Number', 'value': 14.0}},
        'attributes': ['value']
    }
    node = from_dict(data)
    assert isinstance(node, IndicatorWithParams)
    assert node.indicator == 'rsi'
    assert isinstance(node.params['period'], Number)
    assert node.params['period'].value == 14.0
    # Comparison
    data = {
        'type': 'Comparison',
        'operator': '>',
        'left': {'type': 'Number', 'value': 5.0},
        'right': {'type': 'Number', 'value': 3.0}
    }
    node = from_dict(data)
    assert isinstance(node, Comparison)
    assert node.operator == '>'
    assert isinstance(node.left, Number)
    assert node.left.value == 5.0
    assert isinstance(node.right, Number)
    assert node.right.value == 3.0
    # MultiComparison
    data = {
        'type': 'MultiComparison',
        'operators': ['<', '<='],
        'operands': [
            {'type': 'Number', 'value': 1.0},
            {'type': 'Var', 'name': 'x'},
            {'type': 'Number', 'value': 10.0}
        ]
    }
    node = from_dict(data)
    assert isinstance(node, MultiComparison)
    assert node.operators == ['<', '<=']
    assert len(node.operands) == 3
    assert isinstance(node.operands[0], Number)
    assert node.operands[0].value == 1.0
    assert isinstance(node.operands[1], Var)
    assert node.operands[1].name == 'x'
    assert isinstance(node.operands[2], Number)
    assert node.operands[2].value == 10.0
    # LogicalBinOp
    data = {
        'type': 'LogicalBinOp',
        'operator': 'and',
        'left': {'type': 'Number', 'value': 1.0},
        'right': {'type': 'Number', 'value': 0.0}
    }
    node = from_dict(data)
    assert isinstance(node, LogicalBinOp)
    assert node.operator == 'and'
    assert isinstance(node.left, Number)
    assert node.left.value == 1.0
    assert isinstance(node.right, Number)
    assert node.right.value == 0.0
    # LogicalNot
    data = {
        'type': 'LogicalNot',
        'operand': {'type': 'Number', 'value': 1.0}
    }
    node = from_dict(data)
    assert isinstance(node, LogicalNot)
    assert isinstance(node.operand, Number)
    assert node.operand.value == 1.0
    # Let
    data = {
        'type': 'Let',
        'var': 'x',
        'value': {'type': 'Number', 'value': 5.0},
        'body': {'type': 'Number', 'value': 10.0}
    }
    node = from_dict(data)
    assert isinstance(node, Let)
    assert node.var == 'x'
    assert isinstance(node.value, Number)
    assert node.value.value == 5.0
    assert isinstance(node.body, Number)
    assert node.body.value == 10.0
    # HistoricalAccess (with IndicatorAccess)
    data = {
        'type': 'HistoricalAccess',
        'expr': {
            'type': 'IndicatorAccess',
            'indicator': 'close',
            'attributes': []
        },
        'offset': 2
    }
    node = from_dict(data)
    assert isinstance(node, HistoricalAccess)
    assert node.offset == 2
    assert isinstance(node.expr, IndicatorAccess)
    assert node.expr.indicator == 'close'
    # HistoricalAccess (with IndicatorWithParams)
    data = {
        'type': 'HistoricalAccess',
        'expr': {
            'type': 'IndicatorWithParams',
            'indicator': 'rsi',
            'params': {'period': {'type': 'Number', 'value': 14.0}},
            'attributes': []
        },
        'offset': 2
    }
    node = from_dict(data)
    assert isinstance(node, HistoricalAccess)
    assert node.offset == 2
    assert isinstance(node.expr, IndicatorWithParams)
    assert node.expr.indicator == 'rsi'
    # Rising
    data = {
        'type': 'Rising',
        'expr': {
            'type': 'IndicatorAccess',
            'indicator': 'close',
            'attributes': []
        },
        'n': 5
    }
    node = from_dict(data)
    assert isinstance(node, Rising)
    assert node.n == 5
    assert isinstance(node.expr, IndicatorAccess)
    assert node.expr.indicator == 'close'
    # Falling
    data = {
        'type': 'Falling',
        'expr': {
            'type': 'IndicatorAccess',
            'indicator': 'close',
            'attributes': []
        },
        'n': 3
    }
    node = from_dict(data)
    assert isinstance(node, Falling)
    assert node.n == 3
    assert isinstance(node.expr, IndicatorAccess)
    assert node.expr.indicator == 'close'
    # Add
    data = {
        'type': 'Add',
        'left': {'type': 'Number', 'value': 1.0},
        'right': {'type': 'Number', 'value': 2.0}
    }
    node = from_dict(data)
    assert isinstance(node, Add)
    assert isinstance(node.left, Number)
    assert node.left.value == 1.0
    assert isinstance(node.right, Number)
    assert node.right.value == 2.0
    # Sub
    data = {
        'type': 'Sub',
        'left': {'type': 'Number', 'value': 5.0},
        'right': {'type': 'Number', 'value': 3.0}
    }
    node = from_dict(data)
    assert isinstance(node, Sub)
    assert isinstance(node.left, Number)
    assert node.left.value == 5.0
    assert isinstance(node.right, Number)
    assert node.right.value == 3.0
    # Mul
    data = {
        'type': 'Mul',
        'left': {'type': 'Number', 'value': 2.0},
        'right': {'type': 'Number', 'value': 3.0}
    }
    node = from_dict(data)
    assert isinstance(node, Mul)
    assert isinstance(node.left, Number)
    assert node.left.value == 2.0
    assert isinstance(node.right, Number)
    assert node.right.value == 3.0
    # Div
    data = {
        'type': 'Div',
        'left': {'type': 'Number', 'value': 10.0},
        'right': {'type': 'Number', 'value': 2.0}
    }
    node = from_dict(data)
    assert isinstance(node, Div)
    assert isinstance(node.left, Number)
    assert node.left.value == 10.0
    assert isinstance(node.right, Number)
    assert node.right.value == 2.0
    # Mod
    data = {
        'type': 'Mod',
        'left': {'type': 'Number', 'value': 10.0},
        'right': {'type': 'Number', 'value': 3.0}
    }
    node = from_dict(data)
    assert isinstance(node, Mod)
    assert isinstance(node.left, Number)
    assert node.left.value == 10.0
    assert isinstance(node.right, Number)
    assert node.right.value == 3.0
    # Pow
    data = {
        'type': 'Pow',
        'left': {'type': 'Number', 'value': 2.0},
        'right': {'type': 'Number', 'value': 3.0}
    }
    node = from_dict(data)
    assert isinstance(node, Pow)
    assert isinstance(node.left, Number)
    assert node.left.value == 2.0
    assert isinstance(node.right, Number)
    assert node.right.value == 3.0
    # UnaryMinus
    data = {
        'type': 'UnaryMinus',
        'operand': {'type': 'Number', 'value': 5.0}
    }
    node = from_dict(data)
    assert isinstance(node, UnaryMinus)
    assert isinstance(node.operand, Number)
    assert node.operand.value == 5.0
    # Unknown type
    with pytest.raises(ValueError, match='Unknown AST node type'):
        from_dict({'type': 'Unknown'})


@pytest.mark.unit
@pytest.mark.ast
def test_round_trip_through_parser() -> None:
    """Test that parsing, serializing, and deserializing yields
    equivalent AST.
    """
    code = 'let x = rsi(period=14).value in x > 70'
    parser = Parser()
    original_ast = parser.parse(code)
    serialized = original_ast.to_dict()
    reconstructed = from_dict(serialized)
    # Compare by re-serializing both and comparing dicts
    original_serialized = original_ast.to_dict()
    reconstructed_serialized = reconstructed.to_dict()
    assert original_serialized == reconstructed_serialized


@pytest.mark.unit
@pytest.mark.ast
def test_complex_nested_serialization() -> None:
    """Test serialization of a deeply nested AST."""
    # Construct: (1 + 2) * (3 - 4)
    node = Mul(
        left=Add(left=Number(value=1.0), right=Number(value=2.0)),
        right=Sub(left=Number(value=3.0), right=Number(value=4.0))
    )
    data = node.to_dict()
    reconstructed = from_dict(data)
    assert isinstance(reconstructed, Mul)
    assert isinstance(reconstructed.left, Add)
    assert isinstance(reconstructed.left.left, Number)
    assert reconstructed.left.left.value == 1.0
    assert isinstance(reconstructed.left.right, Number)
    assert reconstructed.left.right.value == 2.0
    assert isinstance(reconstructed.right, Sub)
    assert isinstance(reconstructed.right.left, Number)
    assert reconstructed.right.left.value == 3.0
    assert isinstance(reconstructed.right.right, Number)
    assert reconstructed.right.right.value == 4.0


@pytest.mark.unit
@pytest.mark.ast
def test_round_trip_with_empty_containers() -> None:
    """Test serialization round-trip with empty lists and dicts."""
    # IndicatorAccess with empty attributes
    node = IndicatorAccess(indicator='close', attributes=[])
    data = node.to_dict()
    reconstructed = IndicatorAccess.from_dict(data)
    assert reconstructed.attributes == []
    # IndicatorWithParams with empty params and attributes
    node_with_params = IndicatorWithParams(
        indicator='rsi',
        params={},
        attributes=[]
    )
    data = node_with_params.to_dict()
    reconstructed_with_params = IndicatorWithParams.from_dict(data)
    assert reconstructed_with_params.params == {}
    assert reconstructed_with_params.attributes == []
    # MultiComparison with empty operators and operands
    node_multi = MultiComparison(operators=[], operands=[])
    data = node_multi.to_dict()
    reconstructed_multi = MultiComparison.from_dict(data)
    assert reconstructed_multi.operators == []
    assert reconstructed_multi.operands == []
