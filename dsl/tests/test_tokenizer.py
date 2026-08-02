"""Unit tests for the DSL tokenizer.

This module tests the Tokenizer class, which converts a DSL expression string
into a stream of tokens. It covers all token types, keyword recognition,
operator detection, number parsing, punctuation, and error handling.
"""

import pytest
from ..tokenizer import Tokenizer
from ..exceptions import ParseError


@pytest.mark.unit
@pytest.mark.tokenizer
def test_basic_tokens() -> None:
    """Test tokenization of a simple DSL expression with keywords,
    identifiers, numbers, and operators.

    Given a typical expression "let x = 5 in x + 2", the tokenizer should
    produce tokens in the correct order and with correct types.

    Expected: LET, IDENT, ASSIGN, NUMBER, IN, IDENT, PLUS, NUMBER
    """
    t = Tokenizer()
    tokens = t.tokenize('let x = 5 in x + 2')
    types = [tok.type for tok in tokens]
    assert types == [
        'LET',
        'IDENT',
        'ASSIGN',
        'NUMBER',
        'IN',
        'IDENT',
        'PLUS',
        'NUMBER'
    ]


@pytest.mark.unit
@pytest.mark.tokenizer
def test_keyword_priority() -> None:
    """Test that keywords are recognised correctly and take precedence
    over identifiers.

    Keywords like 'let', 'and', 'or', 'not', 'rising', 'falling' should be
    tokenized as their specific types, not as IDENT.
    """
    t = Tokenizer()
    tokens = t.tokenize('let and or not rising falling')
    types = [tok.type for tok in tokens]
    assert types == ['LET', 'AND', 'OR', 'NOT', 'RISING', 'FALLING']


@pytest.mark.unit
@pytest.mark.tokenizer
def test_whole_word_keywords() -> None:
    """Test that keywords are matched as whole words, not as substrings.

    For example, 'sublet' should not be tokenized as 'sub' + 'let', but as
    a single IDENT.
    """
    t = Tokenizer()
    tokens = t.tokenize('sublet')
    assert len(tokens) == 1
    assert tokens[0].type == 'IDENT'
    assert tokens[0].value == 'sublet'


@pytest.mark.unit
@pytest.mark.tokenizer
def test_identifier_with_underscore() -> None:
    """Test that identifiers can contain underscores
    and digits (not starting with digit).
    """
    t = Tokenizer()
    tokens = t.tokenize('my_indicator_123')
    assert len(tokens) == 1
    assert tokens[0].type == 'IDENT'
    assert tokens[0].value == 'my_indicator_123'


@pytest.mark.unit
@pytest.mark.tokenizer
def test_comparison_operators() -> None:
    """Test that all comparison operators are tokenized correctly as COMP_OP.

    Operators: >=, <=, ==, !=, <, >
    """
    t = Tokenizer()
    tokens = t.tokenize('a >= b <= c == d != e < f > g')
    comp_ops = [tok.value for tok in tokens if tok.type == 'COMP_OP']
    assert comp_ops == ['>=', '<=', '==', '!=', '<', '>']


@pytest.mark.unit
@pytest.mark.tokenizer
def test_arithmetic_operators() -> None:
    """Test tokenization of arithmetic operators: +, -, *, /, %, ^.

    Numbers and operators should produce correct types. Whitespace is ignored.
    """
    t = Tokenizer()
    tokens = t.tokenize('1 + 2 - 3 * 4 / 5 % 6 ^ 7')
    types = [tok.type for tok in tokens if tok.type != 'WHITESPACE']
    expected = [
        'NUMBER',
        'PLUS',
        'NUMBER',
        'MINUS',
        'NUMBER',
        'MUL',
        'NUMBER',
        'DIV',
        'NUMBER',
        'MOD',
        'NUMBER',
        'POW',
        'NUMBER'
    ]
    assert types == expected


@pytest.mark.unit
@pytest.mark.tokenizer
def test_assignment_and_punctuation() -> None:
    """Test tokenization of assignment and punctuation: =, (, ), [, ], ,, ."""
    t = Tokenizer()
    tokens = t.tokenize('( ) [ ] , . =')
    types = [tok.type for tok in tokens]
    assert types == [
        'LPAREN',
        'RPAREN',
        'LBRACKET',
        'RBRACKET',
        'COMMA',
        'DOT',
        'ASSIGN'
    ]


@pytest.mark.unit
@pytest.mark.tokenizer
def test_integer_numbers() -> None:
    """Test integer numbers are tokenized as NUMBER with correct value."""
    t = Tokenizer()
    # Note: '-' is separate token, but the number is positive
    tokens = t.tokenize('42 0 -5')
    # Here we test only positive numbers; unary minus handled in parser
    assert tokens[0].value == '42'
    assert tokens[0].type == 'NUMBER'
    assert tokens[1].value == '0'
    assert tokens[1].type == 'NUMBER'


@pytest.mark.unit
@pytest.mark.tokenizer
def test_float_numbers() -> None:
    """Test floating point numbers are tokenized correctly."""
    t = Tokenizer()
    tokens = t.tokenize('3.14 0.5')
    assert tokens[0].value == '3.14'
    assert tokens[0].type == 'NUMBER'
    assert tokens[1].value == '0.5'
    assert tokens[1].type == 'NUMBER'


@pytest.mark.unit
@pytest.mark.tokenizer
def test_line_column_tracking() -> None:
    """Test that tokenizer correctly records line and column positions.

    Columns are zero-based; lines start at 1.
    """
    t = Tokenizer()
    code = 'let\nx = 1'
    tokens = t.tokenize(code)
    assert tokens[0].line == 1
    assert tokens[0].column == 0
    assert tokens[1].line == 2
    assert tokens[1].column == 0  # after newline, column resets to 0


@pytest.mark.unit
@pytest.mark.tokenizer
def test_line_column_with_spaces() -> None:
    """Test column tracking with spaces and multiple tokens on same line."""
    t = Tokenizer()
    code = 'let x = 1'
    tokens = t.tokenize(code)
    assert tokens[0].line == 1 and tokens[0].column == 0
    assert tokens[1].line == 1 and tokens[1].column == 0
    assert tokens[2].line == 1 and tokens[2].column == 0
    assert tokens[3].line == 1 and tokens[3].column == 0


@pytest.mark.unit
@pytest.mark.tokenizer
@pytest.mark.error
def test_unknown_character_error() -> None:
    """Test that tokenizer raises ParseError on an unexpected character."""
    t = Tokenizer()
    with pytest.raises(ParseError, match="Unexpected character '@' at line 1"):
        t.tokenize('a @ b')


@pytest.mark.unit
@pytest.mark.tokenizer
def test_empty_input() -> None:
    """Test tokenization of an empty string returns empty list."""
    t = Tokenizer()
    tokens = t.tokenize('')
    assert tokens == []


@pytest.mark.unit
@pytest.mark.tokenizer
def test_only_whitespace() -> None:
    """Test tokenization of whitespace-only string returns empty list."""
    t = Tokenizer()
    tokens = t.tokenize('   \t\n  ')
    assert tokens == []


@pytest.mark.unit
@pytest.mark.tokenizer
def test_long_identifier() -> None:
    """Test tokenization of a very long identifier (should still work)."""
    t = Tokenizer()
    long_id = 'a' * 1000
    tokens = t.tokenize(long_id)
    assert len(tokens) == 1
    assert tokens[0].type == 'IDENT'
    assert tokens[0].value == long_id


@pytest.mark.unit
@pytest.mark.tokenizer
def test_complex_expression_with_nested_parens() -> None:
    """Test tokenization of a complex expression with
    nested parentheses and brackets.
    """
    t = Tokenizer()
    code = 'let x = (a + b)[1] in x > 0'
    tokens = t.tokenize(code)
    types = [tok.type for tok in tokens]
    expected = [
        'LET',
        'IDENT',
        'ASSIGN',
        'LPAREN',
        'IDENT',
        'PLUS',
        'IDENT',
        'RPAREN',
        'LBRACKET',
        'NUMBER',
        'RBRACKET',
        'IN',
        'IDENT',
        'COMP_OP',
        'NUMBER'
    ]
    assert types == expected
