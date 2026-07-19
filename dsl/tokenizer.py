"""Lexer for DSL. Recognizes all operators and keywords."""

import re
from typing import NamedTuple

from .exceptions import ParseError


class Token(NamedTuple):
    """Represents a lexical token with type, value, and position."""

    type: str
    value: str
    line: int
    column: int


class Tokenizer:
    """Lexical analyzer for DSL expressions.

    Converts input string into a sequence of tokens. Supports numbers,
    identifiers, keywords (let, in, and, or, not, rising, falling), comparison
    operators, arithmetic operators, parentheses, brackets, comma, dot
    and assignment.
    """

    def __init__(self) -> None:
        """Initialize the tokenizer with regex patterns for all token types.

        Order is important: longer operators (>=, <=, ==, !=) must come before
        shorter ones (<, >, =) to avoid incorrect matching.
        Keywords must come before IDENT to be recognized as specific
        token types.
        """
        self.spec = [
            # Keywords (must come before IDENT)
            # Added \b at the start to ensure whole-word matching only
            ('LET',     r'\blet\b'),
            ('IN',      r'\bin\b'),
            ('AND',     r'\band\b'),
            ('OR',      r'\bor\b'),
            ('NOT',     r'\bnot\b'),
            ('RISING',  r'\brising\b'),
            ('FALLING', r'\bfalling\b'),
            # Identifier (must come after keywords)
            ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            # Number
            ('NUMBER', r'\d+(\.\d+)?'),
            # Comparison operators (longer first)
            ('GE', r'>='),
            ('LE', r'<='),
            ('EQ', r'=='),
            ('NE', r'!='),
            ('LT', r'<'),
            ('GT', r'>'),
            # Arithmetic operators
            ('PLUS', r'\+'),
            ('MINUS', r'-'),
            ('MUL', r'\*'),
            ('DIV', r'/'),
            ('MOD', r'%'),
            ('POW', r'\^'),
            # Punctuation
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACKET', r'\['),
            ('RBRACKET', r'\]'),
            ('COMMA', r','),
            ('ASSIGN', r'='),
            ('DOT', r'\.'),
            # Whitespace and unknown
            ('WHITESPACE', r'\s+'),
            ('UNKNOWN', r'.'),
        ]
        self.regex = re.compile(
            '|'.join(
                f'(?P<{name}>{pattern})' for name,
                pattern in self.spec
            )
        )

    def tokenize(self, code: str) -> list[Token]:
        """Convert the input string into a list of tokens.

        Args:
            code: The source code to tokenize.

        Returns:
            A list of Token objects.

        Raises:
            ParseError: If an unexpected character is encountered.

        """
        tokens: list[Token] = []
        line = 1
        pos = 0
        for mo in self.regex.finditer(code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'WHITESPACE':
                line += value.count('\n')
                pos = mo.end()
                continue
            if kind == 'UNKNOWN' or kind is None:
                raise ParseError(
                    f"Unexpected character '{value}' at line {line}"
                )
            # Normalize comparison operators to a single type
            if kind in ('LT', 'GT', 'LE', 'GE', 'EQ', 'NE'):
                kind = 'COMP_OP'
            tokens.append(Token(kind, value, line, mo.start() - pos))
        return tokens
