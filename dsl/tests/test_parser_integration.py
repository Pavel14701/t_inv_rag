"""Integration tests for the full DSL pipeline.

These tests verify that expressions produce the expected boolean result,
using a mock context with predefined indicator values.
This approach decouples tests from the internal AST structure,
allowing the implementation to change without breaking tests.
"""

import pytest
from dsl.evaluate import evaluate_dsl
from dsl.exceptions import ParseError, EvaluationError
from dsl.parser import Parser


@pytest.mark.integration
@pytest.mark.parser
@pytest.mark.with_providers
@pytest.mark.parametrize(
    'code,expected',
    [
        ('1 + 2', True),
        ('5 - 5', False),
        ('2 * 3', True),
        ('10 / 2', True),
        ('0', False),
        ('0.0', False),
        ('1', True),
        ('-1', True),

        ('5 > 3', True),
        ('5 < 3', False),
        ('5 == 5', True),
        ('5 != 5', False),
        ('5 <= 5', True),
        ('5 >= 5', True),

        ('1 < 5 < 10', True),
        ('10 < 5 < 1', False),
        ('5 == 5 >= 5', True),

        ('1 and 1', True),
        ('1 and 0', False),
        ('0 or 1', True),
        ('not 0', True),
        ('not 1', False),
        ('not (5 < 3)', True),

        ('1 + 2 * 3', True),
        ('(1 + 2) * 3', True),
        ('2 ^ 3 ^ 2', True),
        ('-2 ^ 3', True),

        ('close', True),
        ('close > 50', True),
        ('close < 50', False),
        ('close == 100', False),

        ('rsi(period=14).value > 60', True),
        ('rsi(period=14).value < 50', False),

        ('macd(fast=12, slow=26).line > macd(fast=12, slow=26).signal', True),
        ('macd.line > macd.signal', True),

        ('close[1] > 90', True),
        ('close[1] < 80', False),
        ('close[0] == 100', False),

        ('rising(close, 3)', True),
        ('falling(volume, 3)', False),

        ('let x = 5 in x > 3', True),
        ('let x = close in x > 50', True),
        ('let x = close - 100 in x > 0', True),
        ('let x = 5 in let y = x + 1 in y == 6', True),

        ('close > 100 and volume > 500000', True),
        ('close > 100 and volume < 500000', False),
        ('(close > 100) or (volume < 500000)', True),

        ('let price = close in rising(close, 2) and price > 90', True),

        ('0 and 5', False),
        ('5 and 0', False),
        ('0 or 5', True),
        ('5 or 0', True),
    ]
)
def test_dsl_evaluation(mock_context, code, expected):
    """Test that DSL expressions evaluate to the expected boolean result."""
    assert evaluate_dsl(code, mock_context) == expected


# -----------------------------------------------------------------------------
# Error cases
# -----------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parser
@pytest.mark.with_providers
@pytest.mark.error
@pytest.mark.parametrize(
    'code,exception_type,error_substring',
    [
        ('1 +', ParseError, 'Unexpected EOF'),
        (')', ParseError, 'Unexpected token'),
        ('(1 + 2', ParseError, 'Expected RPAREN'),
        ('close[1', ParseError, 'Expected RBRACKET'),
        ('rising(close 5)', ParseError, 'Expected COMMA'),
        ('rsi.', ParseError, 'Expected IDENT'),
        ('(close + 1)[1]', ParseError, 'Unexpected token'),
        ('1 / 0', EvaluationError, 'Division by zero'),
        ('1 % 0', EvaluationError, 'Modulo by zero'),
        (
            'unknown_indicator',
            ValueError,
            'Unknown indicator: unknown_indicator'
        ),
        (
            'rising(close + 1, 3)',
            EvaluationError,
            r'rising\(\) expects an indicator expression'
        ),
        (
            'let x = close in rising(x, 3)',
            EvaluationError,
            r'rising\(\) expects an indicator expression'
        ),
    ]
)
def test_dsl_errors(mock_context, code, exception_type, error_substring):
    """Test that DSL expressions raise the expected exception."""
    with pytest.raises(exception_type, match=error_substring):
        evaluate_dsl(code, mock_context)


# -----------------------------------------------------------------------------
# Additional AST structure tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parser
def test_ast_let_shadowing():
    """Test that let shadowing produces correct AST structure."""
    from dsl.ast import Let, LogicalBinOp, Var, Number, Comparison
    code = 'let x = 5 in (let x = 10 in x) and x == 5'
    parser = Parser()
    ast = parser.parse(code)
    assert isinstance(ast, Let)
    assert ast.var == 'x'
    body = ast.body
    assert isinstance(body, LogicalBinOp)
    assert body.operator == 'and'
    left = body.left
    assert isinstance(left, Let)
    assert left.var == 'x'
    assert isinstance(left.body, Var)
    assert left.body.name == 'x'
    right = body.right
    assert isinstance(right, Comparison)
    assert right.operator == '=='
    assert isinstance(right.left, Var)
    assert right.left.name == 'x'
    assert isinstance(right.right, Number)
    assert right.right.value == 5.0


@pytest.mark.unit
@pytest.mark.parser
def test_ast_indicator_with_arithmetic_params():
    """Test that indicator parameters can be arithmetic expressions."""
    parser = Parser()
    ast = parser.parse('rsi(period=14+1).value')
    from dsl.ast import IndicatorWithParams, Add, Number
    assert isinstance(ast, IndicatorWithParams)
    assert ast.indicator == 'rsi'
    assert 'period' in ast.params
    param = ast.params['period']
    assert isinstance(param, Add)
    assert isinstance(param.left, Number)
    assert param.left.value == 14.0
    assert isinstance(param.right, Number)
    assert param.right.value == 1.0


@pytest.mark.unit
@pytest.mark.parser
def test_ast_historical_with_params_and_attrs():
    """Test historical access with parameters and attributes."""
    parser = Parser()
    ast = parser.parse('rsi(period=14).value[2]')
    from dsl.ast import HistoricalAccess, IndicatorWithParams
    assert isinstance(ast, HistoricalAccess)
    assert ast.offset == 2
    expr = ast.expr
    assert isinstance(expr, IndicatorWithParams)
    assert expr.indicator == 'rsi'
    assert expr.attributes == ['value']


@pytest.mark.unit
@pytest.mark.parser
def test_ast_rising_with_params():
    """Test rising with indicator that has parameters."""
    parser = Parser()
    ast = parser.parse('rising(rsi(period=14).value, 3)')
    from dsl.ast import Rising, IndicatorWithParams
    assert isinstance(ast, Rising)
    assert ast.n == 3
    expr = ast.expr
    assert isinstance(expr, IndicatorWithParams)
    assert expr.indicator == 'rsi'
    assert expr.attributes == ['value']


@pytest.mark.unit
@pytest.mark.parser
def test_ast_falling_with_params():
    """Test falling with indicator that has parameters."""
    parser = Parser()
    ast = parser.parse('falling(rsi(period=14).value, 3)')
    from dsl.ast import Falling, IndicatorWithParams
    assert isinstance(ast, Falling)
    assert ast.n == 3
    expr = ast.expr
    assert isinstance(expr, IndicatorWithParams)
    assert expr.indicator == 'rsi'
    assert expr.attributes == ['value']


@pytest.mark.unit
@pytest.mark.parser
def test_ast_multi_comparison_with_identifiers():
    """Test that chained comparison produces correct AST."""
    parser = Parser()
    ast = parser.parse('a < b <= c')
    from dsl.ast import MultiComparison, IndicatorAccess
    assert isinstance(ast, MultiComparison)
    assert ast.operators == ['<', '<=']
    assert len(ast.operands) == 3
    assert all(isinstance(op, IndicatorAccess) for op in ast.operands)


@pytest.mark.unit
@pytest.mark.parser
def test_ast_precedence_with_parentheses():
    """Test that parentheses override precedence correctly."""
    parser = Parser()
    ast = parser.parse('(1 + 2) * 3')
    from dsl.ast import Mul, Add, Number
    assert isinstance(ast, Mul)
    left = ast.left
    assert isinstance(left, Add)
    assert isinstance(left.left, Number)
    assert left.left.value == 1.0
    assert isinstance(left.right, Number)
    assert left.right.value == 2.0
    assert isinstance(ast.right, Number)
    assert ast.right.value == 3.0


@pytest.mark.unit
@pytest.mark.tokenizer
def test_tokenizer_unknown_character():
    """Test that tokenizer raises ParseError on unknown character."""
    from dsl.tokenizer import Tokenizer
    t = Tokenizer()
    with pytest.raises(ParseError, match="Unexpected character '@' at line 1"):
        t.tokenize('a @ b')


@pytest.mark.unit
@pytest.mark.parser
def test_error_line_column():
    """Test that parse errors report some message."""
    parser = Parser()
    with pytest.raises(ParseError) as excinfo:
        parser.parse('1 +')
    assert 'Unexpected EOF' in str(excinfo.value)
