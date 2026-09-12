"""Recursive descent parser for DSL with all operators."""
from .ast import (
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
from .exceptions import ParseError
from .tokenizer import Token, Tokenizer


class Parser:
    """Recursive descent parser for DSL expressions.

    This parser implements a top-down parser with precedence climbing
    for arithmetic and comparison operators, and supports all DSL constructs.
    It also tracks `let`-bound variables so that references to them
    inside the body produce `Var` nodes instead of indicator accesses.

    Attributes:
        tokens: List of tokens from the tokenizer.
        pos: Current position in the token list.

    """

    def __init__(self) -> None:
        """Initialize the parser with an empty token list."""
        self.tokens: list[Token] = []
        self.pos = 0
        # Stack of sets representing active let-bound variable names.
        # Each element corresponds to a nested scope.
        self._let_stack: list[set[str]] = []

    def parse(self, code: str) -> ASTNode:
        """Parse a DSL expression string into an AST.

        Args:
            code: The DSL expression to parse.

        Returns:
            The root AST node.

        Raises:
            ParseError: If the input contains syntax errors.

        """
        tokenizer = Tokenizer()
        self.tokens = tokenizer.tokenize(code)
        self.pos = 0
        result = self._expression()
        if self.pos < len(self.tokens):
            raise ParseError(
                f"Unexpected token at end: {self.tokens[self.pos].value}"
            )
        return result

    def _peek(self) -> Token | None:
        """Return the current token without consuming it, or None if EOF."""
        return None if self.pos >= len(self.tokens) else self.tokens[self.pos]

    def _next(self) -> Token:
        """Consume and return the next token.

        Raises:
            ParseError: If at the end of input.

        """
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of input")
        self.pos += 1
        return tok

    def _match(self, expected_type: str) -> Token:
        """Check if the next token has the expected type and consume it.

        Args:
            expected_type: The expected token type string.

        Returns:
            The matched token.

        Raises:
            ParseError: If the token type does not match or at end of input.

        """
        tok = self._peek()
        if tok is None or tok.type != expected_type:
            raise ParseError(
                f'Expected {expected_type}, got {tok.value if tok else "EOF"}'
            )
        return self._next()

    # ---------- Grammar rules ----------

    def _expression(self) -> ASTNode:
        """Parse an expression.

        Grammar:
            expression = let_expr | or_expr

        Returns:
            The parsed AST node (Let or logical expression).

        """
        tok = self._peek()
        return (
            self._let_expr() if tok and tok.type == "LET"
            else self._or_expr()
        )

    def _let_expr(self) -> ASTNode:
        """Parse a let binding expression.

        Grammar:
            let_expr = 'let' IDENT '=' or_expr 'in' expression

        Returns:
            A Let node with the bound variable and body.

        """
        self._match("LET")
        ident = self._match("IDENT")
        var_name = ident.value
        self._match("ASSIGN")

        # Create a new scope with this variable
        new_scope = {var_name}
        self._let_stack.append(new_scope)

        value = self._or_expr()
        self._match("IN")
        body = self._expression()

        # Remove the scope after parsing the body
        self._let_stack.pop()

        return Let(var=var_name, value=value, body=body)

    def _or_expr(self) -> ASTNode:
        """Parse an OR expression.

        Grammar:
            or_expr = and_expr ('or' and_expr)*

        Returns:
            LogicalBinOp node with 'or' operator, left-associative.

        """
        node = self._and_expr()
        tok = self._peek()
        while tok and tok.type == "OR":
            self._next()
            right = self._and_expr()
            node = LogicalBinOp(operator="or", left=node, right=right)
            tok = self._peek()
        return node

    def _and_expr(self) -> ASTNode:
        """Parse an AND expression.

        Grammar:
            and_expr = not_expr ('and' not_expr)*

        Returns:
            LogicalBinOp node with 'and' operator, left-associative.

        """
        node = self._not_expr()
        tok = self._peek()
        while tok and tok.type == "AND":
            self._next()
            right = self._not_expr()
            node = LogicalBinOp(operator="and", left=node, right=right)
            tok = self._peek()
        return node

    def _not_expr(self) -> ASTNode:
        """Parse a NOT expression.

        Grammar:
            not_expr = 'not' not_expr | comparison

        Returns:
            LogicalNot node if NOT is present, otherwise a comparison node.

        """
        tok = self._peek()
        if tok and tok.type == "NOT":
            self._next()
            node = self._not_expr()
            return LogicalNot(operand=node)
        return self._comparison()

    def _comparison(self) -> ASTNode:
        """Parse a comparison expression.

        Grammar:
            comparison = arith_expr (comp_op arith_expr)*

        Returns:
            Comparison node if a single comparison, MultiComparison if chained,
            otherwise the left arithmetic expression.

        """
        left = self._arith_expr()
        tok = self._peek()
        if tok and tok.type == "COMP_OP":
            ops = []
            operands = [left]
            while tok and tok.type == "COMP_OP":
                op_tok = self._next()
                ops.append(op_tok.value)
                right = self._arith_expr()
                operands.append(right)
                tok = self._peek()
            if len(ops) == 1:
                return Comparison(
                    operator=ops[0],
                    left=operands[0],
                    right=operands[1]
                )
            return MultiComparison(operators=ops, operands=operands)
        return left

    def _arith_expr(self) -> ASTNode:
        """Parse an arithmetic expression.

        Grammar:
            arith_expr = term (('+' | '-') term)*

        Returns:
            AST node with addition or subtraction operations.

        """
        node = self._term()
        tok = self._peek()
        while tok and tok.type in ("PLUS", "MINUS"):
            op_tok = self._next()
            right = self._term()
            if op_tok.type == "PLUS":
                node = Add(left=node, right=right)
            else:
                node = Sub(left=node, right=right)
            tok = self._peek()
        return node

    def _term(self) -> ASTNode:
        """Parse a term expression.

        Grammar:
            term = factor (('*' | '/' | '%') factor)*

        Returns:
            AST node with multiplication, division, or modulo operations.

        """
        node = self._factor()
        tok = self._peek()
        while tok and tok.type in ("MUL", "DIV", "MOD"):
            op_tok = self._next()
            right = self._factor()
            if op_tok.type == "MUL":
                node = Mul(left=node, right=right)
            elif op_tok.type == "DIV":
                node = Div(left=node, right=right)
            else:
                node = Mod(left=node, right=right)
            tok = self._peek()
        return node

    def _factor(self) -> ASTNode:
        """Parse a factor expression.

        Grammar:
            factor = unary ('^' unary)?  # right-associative

        Returns:
            The parsed AST node (Unary or Pow).

        """
        node = self._unary()
        tok = self._peek()
        if tok and tok.type == "POW":
            self._next()
            right = self._factor()
            return Pow(left=node, right=right)
        return node

    def _unary(self) -> ASTNode:
        """Parse a unary expression.

        Grammar:
            unary = ('-')? atom

        Returns:
            UnaryMinus node if unary minus is present, otherwise the atom node.

        """
        tok = self._peek()
        if tok and tok.type == "MINUS":
            self._next()
            operand = self._unary()
            return UnaryMinus(operand=operand)
        return self._atom()

    def _atom(self) -> ASTNode:
        """Parse an atomic expression.

        Grammar:
            atom = NUMBER
                | IDENT (-> Var if let-bound, else indicator access)
                | '(' expression ')'
                | RISING '(' expression ',' NUMBER ')'
                | FALLING '(' expression ',' NUMBER ')'

        Returns:
            The parsed atom AST node.

        """
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected EOF")
        if tok.type == "NUMBER":
            self._next()
            return Number(value=float(tok.value))
        if tok.type == "IDENT":
            # If the identifier is a let-bound variable, emit a Var node.
            if self._is_let_var(tok.value):
                self._next()
                return Var(name=tok.value)
            # Otherwise treat it as an indicator access
            # (with optional params/attrs).
            return self._parse_indicator()
        if tok.type == "RISING":
            return self._parse_rising()
        if tok.type == "FALLING":
            return self._parse_falling()
        if tok.type == "LPAREN":
            self._next()
            node = self._expression()
            self._match("RPAREN")
            return node
        raise ParseError(f"Unexpected token: {tok.value}")

    def _is_let_var(self, name: str) -> bool:
        """Check whether the given name refers to an active let-bound variable.

        The lookup walks the stack from the innermost scope outward.
        """
        return any(name in scope for scope in reversed(self._let_stack))

    # ---------- Indicator parsing helpers ----------

    def _parse_rising(self) -> Rising:
        """Parse the rising function: rising(expression, n).

        Returns:
            Rising node.

        Raises:
            ParseError: If the syntax is incorrect.

        """
        self._next()  # consume RISING token
        self._match("LPAREN")
        expr = self._expression()
        self._match("COMMA")
        num_tok = self._match("NUMBER")
        self._match("RPAREN")
        return Rising(expr=expr, n=int(num_tok.value))

    def _parse_falling(self) -> Falling:
        """Parse the falling function: falling(expression, n).

        Returns:
            Falling node.

        Raises:
            ParseError: If the syntax is incorrect.

        """
        self._next()  # consume FALLING token
        self._match("LPAREN")
        expr = self._expression()
        self._match("COMMA")
        num_tok = self._match("NUMBER")
        self._match("RPAREN")
        return Falling(expr=expr, n=int(num_tok.value))

    def _parse_indicator(self) -> ASTNode:
        """Parse an indicator access with optional parameters, attributes,
        and historical offset.

        Grammar:
            indicator = IDENT ['(' param_list ')'] ('.' IDENT)* ['[' NUMBER ']']
            param_list = (IDENT '=' or_expr) (',' IDENT '=' or_expr)*

        Returns:
            IndicatorAccess, IndicatorWithParams, or HistoricalAccess node.

        """  # noqa: E501
        ident_token = self._match("IDENT")
        base_name = ident_token.value
        # Check for parameters enclosed in parentheses
        tok = self._peek()
        if tok and tok.type == "LPAREN":
            return self._parse_indicator_params_and_attrs(base_name)
        # No parameters: collect dotted attributes and optional
        # historical offset
        attrs = []
        tok = self._peek()
        while tok and tok.type == "DOT":
            self._next()
            attr = self._match("IDENT").value
            attrs.append(attr)
            tok = self._peek()
        if tok and tok.type == "LBRACKET":
            offset = self._parse_history_offset()
            expr = IndicatorAccess(indicator=base_name, attributes=attrs)
            return HistoricalAccess(expr=expr, offset=offset)
        return IndicatorAccess(indicator=base_name, attributes=attrs)

    def _parse_indicator_params_and_attrs(self, base_name: str) -> ASTNode:
        """Parse the parenthesised part of an indicator call.

        Handles the opening parenthesis, named parameters, closing parenthesis,
        optional dot-separated attributes and an optional historical offset.
        """
        self._next()  # consume '('
        params = {}
        tok = self._peek()
        if tok and tok.type != "RPAREN":
            while True:
                param_name = self._match("IDENT").value
                self._match("ASSIGN")
                param_expr = self._expression()
                params[param_name] = param_expr
                tok = self._peek()
                if tok and tok.type == "COMMA":
                    self._next()
                    continue
                break
        self._match("RPAREN")
        # Collect optional dotted attributes after the closing paren
        attrs = []
        tok = self._peek()
        while tok and tok.type == "DOT":
            self._next()
            attr = self._match("IDENT").value
            attrs.append(attr)
            tok = self._peek()
        # Optional historical offset
        if tok and tok.type == "LBRACKET":
            offset = self._parse_history_offset()
            expr = IndicatorWithParams(
                indicator=base_name,
                params=params,
                attributes=attrs
            )
            return HistoricalAccess(expr=expr, offset=offset)
        return IndicatorWithParams(
            indicator=base_name,
            params=params,
            attributes=attrs
        )

    def _parse_history_offset(self) -> int:
        """Parse a historical access offset: [NUMBER].

        Returns:
            The integer offset value.

        """
        self._next()  # consume '['
        num_tok = self._match("NUMBER")
        result = int(num_tok.value)
        self._match("RBRACKET")
        return result


def parse(code: str) -> ASTNode:
    """Parse a DSL expression string into an AST.

    This is the main entry point for parsing.

    Args:
        code: The DSL expression string.

    Returns:
        The root AST node.

    Raises:
        ParseError: If the input contains syntax errors.

    """
    parser = Parser()
    return parser.parse(code)
