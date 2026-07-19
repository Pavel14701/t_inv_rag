import pytest
from dsl.tokenizer import Tokenizer, Token
from dsl.exceptions import ParseError

def test_basic_tokens():
    t = Tokenizer()
    tokens = t.tokenize("let x = 5 in x + 2")
    types = [tok.type for tok in tokens]
    assert types == ['LET', 'IDENT', 'ASSIGN', 'NUMBER', 'IN', 'IDENT', 'PLUS', 'NUMBER']

def test_keyword_priority():
    t = Tokenizer()
    tokens = t.tokenize("let and or not rising falling")
    types = [tok.type for tok in tokens]
    assert types == ['LET', 'AND', 'OR', 'NOT', 'RISING', 'FALLING']

def test_whole_word_keywords():
    t = Tokenizer()
    # 'sublet' не должно разбиваться как SUB LET
    tokens = t.tokenize("sublet")
    assert len(tokens) == 1
    assert tokens[0].type == 'IDENT'
    assert tokens[0].value == 'sublet'

def test_comparison_operators():
    t = Tokenizer()
    tokens = t.tokenize("a >= b <= c == d != e < f > g")
    comp_ops = [tok.value for tok in tokens if tok.type == 'COMP_OP']
    assert comp_ops == ['>=', '<=', '==', '!=', '<', '>']

def test_arithmetic_operators():
    t = Tokenizer()
    tokens = t.tokenize("1 + 2 - 3 * 4 / 5 % 6 ^ 7")
    types = [tok.type for tok in tokens if tok.type not in ('WHITESPACE',)]
    expected = ['NUMBER','PLUS','NUMBER','MINUS','NUMBER','MUL','NUMBER','DIV','NUMBER','MOD','NUMBER','POW','NUMBER']
    # whitespace игнорируется
    assert types == expected

def test_numbers():
    t = Tokenizer()
    tokens = t.tokenize("42 3.14")
    assert tokens[0].value == '42'
    assert tokens[0].type == 'NUMBER'
    assert tokens[1].value == '3.14'
    assert tokens[1].type == 'NUMBER'

def test_punctuation():
    t = Tokenizer()
    tokens = t.tokenize("( ) [ ] , . =")
    types = [tok.type for tok in tokens]
    assert types == ['LPAREN','RPAREN','LBRACKET','RBRACKET','COMMA','DOT','ASSIGN']

def test_unknown_character():
    t = Tokenizer()
    with pytest.raises(ParseError, match="Unexpected character"):
        t.tokenize("a @ b")

def test_line_column():
    t = Tokenizer()
    tokens = t.tokenize("let\nx = 1")
    # первый токен 'let' строка 1, столбец 0
    assert tokens[0].line == 1
    assert tokens[0].column == 0
    # второй токен 'x' строка 2
    assert tokens[1].line == 2