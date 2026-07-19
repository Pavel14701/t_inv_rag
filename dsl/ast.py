"""AST node definitions with JSON serialization support.

All nodes are frozen (immutable) and use slots for memory efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------- Basic nodes ----------

@dataclass(frozen=True, slots=True)
class Number:
    """Numeric literal.

    Attributes:
        value: Floating-point number.

    """

    type: str = 'Number'
    value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the Number node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type' and 'value'.

        """
        return {'type': self.type, 'value': self.value}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Number:
        """Reconstruct a Number node from a dictionary.

        Args:
            data: Dictionary containing 'value'.

        Returns:
            Number instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Number(value=data['value'])


@dataclass(frozen=True, slots=True)
class IndicatorAccess:
    """Access to an indicator value without explicit parameters.
    Example: `rsi.value` or `close`.

    Attributes:
        indicator: Name of the indicator.
        attributes: List of attribute names.

    """

    type: str = 'IndicatorAccess'
    indicator: str = ''
    attributes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the IndicatorAccess node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'indicator', and 'attributes'.

        """
        return {
            'type': self.type,
            'indicator': self.indicator,
            'attributes': self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IndicatorAccess:
        """Reconstruct an IndicatorAccess node from a dictionary.

        Args:
            data: Dictionary containing 'indicator'
            and optionally 'attributes'.

        Returns:
            IndicatorAccess instance.

        Raises:
            KeyError: If 'indicator' is missing.

        """
        return IndicatorAccess(
            indicator=data['indicator'],
            attributes=data.get('attributes', [])
        )


@dataclass(frozen=True, slots=True)
class IndicatorWithParams:
    """Access to an indicator with explicit parameters.
    Example: `rsi(period=14).value`.

    Attributes:
        indicator: Name of the indicator.
        params: Mapping from parameter name to expression AST node.
        attributes: List of attribute names.

    """

    type: str = 'IndicatorWithParams'
    indicator: str = ''
    params: dict[str, ASTNode] = field(default_factory=dict)
    attributes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the IndicatorWithParams node to a JSON-serializable
        dictionary.

        Returns:
            A dict with keys 'type', 'indicator', 'params', and 'attributes'.
            'params' is a dict of parameter names to serialized AST nodes.

        """
        return {
            'type': self.type,
            'indicator': self.indicator,
            'params': {k: v.to_dict() for k, v in self.params.items()},
            'attributes': self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IndicatorWithParams:
        """Reconstruct an IndicatorWithParams node from a dictionary.

        Args:
            data: Dictionary containing 'indicator', optionally
            'params' and 'attributes'.

        Returns:
            IndicatorWithParams instance.

        Raises:
            KeyError: If 'indicator' is missing.

        """
        params = {
            k: from_dict(v) for k, v in data.get('params', {}).items()
        }
        return IndicatorWithParams(
            indicator=data['indicator'],
            params=params,
            attributes=data.get('attributes', [])
        )


# ---------- Logical nodes ----------

@dataclass(frozen=True, slots=True)
class LogicalBinOp:
    """Logical AND or OR operation.

    Attributes:
        operator: Either "and" or "or".
        left: Left operand AST node.
        right: Right operand AST node.

    """

    left: ASTNode
    right: ASTNode
    type: str = 'LogicalBinOp'
    operator: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Convert the LogicalBinOp node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'operator', 'left', and 'right'.
            Left and right are serialized recursively.

        """
        return {
            'type': self.type,
            'operator': self.operator,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LogicalBinOp:
        """Reconstruct a LogicalBinOp node from a dictionary.

        Args:
            data: Dictionary containing 'operator', 'left' and 'right'.

        Returns:
            LogicalBinOp instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return LogicalBinOp(
            operator=data['operator'],
            left=from_dict(data['left']),
            right=from_dict(data['right']),
        )


@dataclass(frozen=True, slots=True)
class LogicalNot:
    """Logical NOT operation.

    Attributes:
        operand: Sub-expression to negate.

    """

    operand: ASTNode
    type: str = 'LogicalNot'

    def to_dict(self) -> dict[str, Any]:
        """Convert the LogicalNot node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type' and 'operand'.

        """
        return {'type': self.type, 'operand': self.operand.to_dict()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LogicalNot:
        """Reconstruct a LogicalNot node from a dictionary.

        Args:
            data: Dictionary containing 'operand'.

        Returns:
            LogicalNot instance.

        Raises:
            KeyError: If 'operand' is missing.

        """
        return LogicalNot(operand=from_dict(data['operand']))


# ---------- Comparison nodes ----------

@dataclass(frozen=True, slots=True)
class Comparison:
    """Binary comparison operation.

    Attributes:
        operator: One of '<', '>', '<=', '>=', '==', '!='.
        left: Left operand AST node.
        right: Right operand AST node.

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Comparison'
    operator: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Convert the Comparison node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'operator', 'left', and 'right'.

        """
        return {
            'type': self.type,
            'operator': self.operator,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Comparison:
        """Reconstruct a Comparison node from a dictionary.

        Args:
            data: Dictionary containing 'operator', 'left' and 'right'.

        Returns:
            Comparison instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Comparison(
            operator=data['operator'],
            left=from_dict(data['left']),
            right=from_dict(data['right']),
        )


@dataclass(frozen=True, slots=True)
class MultiComparison:
    """Chained comparison (e.g., a < b <= c).

    Attributes:
        operators: List of comparison operators.
        operands: List of operand AST nodes (length = operators + 1).

    """

    type: str = 'MultiComparison'
    operators: list[str] = field(default_factory=list)
    operands: list[ASTNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the MultiComparison node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'operators', and 'operands'.
            'operands' is a list of serialized AST nodes.

        """
        return {
            'type': self.type,
            'operators': self.operators,
            'operands': [op.to_dict() for op in self.operands],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MultiComparison:
        """Reconstruct a MultiComparison node from a dictionary.

        Args:
            data: Dictionary containing 'operators' and 'operands'.

        Returns:
            MultiComparison instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return MultiComparison(
            operators=data['operators'],
            operands=[from_dict(op) for op in data['operands']],
        )


# ---------- Arithmetic nodes ----------

@dataclass(frozen=True, slots=True)
class Add:
    """Addition operation."""

    left: ASTNode
    right: ASTNode
    type: str = 'Add'

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict with 'type', 'left', 'right'."""
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Add:
        """Reconstruct an Add node from a dict."""
        return Add(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class Sub:
    """Subtraction operation.

    This node represents the subtraction of the right operand from
    the left operand.
    Example: `close - 10` or `rsi.value - 50`.

    Attributes:
        left: Left operand AST node (minuend).
        right: Right operand AST node (subtrahend).

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Sub'

    def to_dict(self) -> dict[str, Any]:
        """Convert the Sub node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'left', and
            'right' (serialized recursively).

        """
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Sub:
        """Reconstruct a Sub node from a dictionary.

        Args:
            data: Dictionary containing 'left' and 'right'.

        Returns:
            Sub instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Sub(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class Mul:
    """Multiplication operation.

    This node represents the multiplication of the left operand by
    the right operand.
    Example: `close * 2` or `rsi.value * 0.5`.

    Attributes:
        left: Left operand AST node (multiplicand).
        right: Right operand AST node (multiplier).

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Mul'

    def to_dict(self) -> dict[str, Any]:
        """Convert the Mul node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'left', and 'right'
            (serialized recursively).

        """
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Mul:
        """Reconstruct a Mul node from a dictionary.

        Args:
            data: Dictionary containing 'left' and 'right'.

        Returns:
            Mul instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Mul(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class Div:
    """Division operation.

    This node represents the division of the left operand by the right operand.
    Example: `close / 10` or `rsi.value / 2`.

    Attributes:
        left: Dividend expression AST node.
        right: Divisor expression AST node.

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Div'

    def to_dict(self) -> dict[str, Any]:
        """Convert the Div node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'left', and 'right'
            (serialized recursively).

        """
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Div:
        """Reconstruct a Div node from a dictionary.

        Args:
            data: Dictionary containing 'left' and 'right'.

        Returns:
            Div instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Div(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class Mod:
    """Modulo operation.

    This node represents the remainder of division of the left operand by
    the right operand.
    Example: `close % 10` or `rsi.value % 5`.

    Attributes:
        left: Dividend expression AST node.
        right: Divisor expression AST node.

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Mod'

    def to_dict(self) -> dict[str, Any]:
        """Convert the Mod node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'left',
            and 'right' (serialized recursively).

        """
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Mod:
        """Reconstruct a Mod node from a dictionary.

        Args:
            data: Dictionary containing 'left' and 'right'.

        Returns:
            Mod instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Mod(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class Pow:
    """Exponentiation operation.

    This node represents raising the left operand to the power of the
    right operand.
    Example: `close ^ 2` or `rsi.value ^ 1.5`.

    Attributes:
        left: Base expression AST node.
        right: Exponent expression AST node.

    """

    left: ASTNode
    right: ASTNode
    type: str = 'Pow'

    def to_dict(self) -> dict[str, Any]:
        """Convert the Pow node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'left', and 'right'
            (serialized recursively).

        """
        return {
            'type': self.type,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Pow:
        """Reconstruct a Pow node from a dictionary.

        Args:
            data: Dictionary containing 'left' and 'right'.

        Returns:
            Pow instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Pow(
            left=from_dict(data['left']),
            right=from_dict(data['right'])
        )


@dataclass(frozen=True, slots=True)
class UnaryMinus:
    """Unary minus operation (negation).

    This node represents the unary negation of an
    expression, e.g., `-rsi.value`.

    Attributes:
        operand: The expression to be negated.

    """

    operand: ASTNode
    type: str = 'UnaryMinus'

    def to_dict(self) -> dict[str, Any]:
        """Convert the UnaryMinus node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type' and 'operand' (serialized recursively).

        """
        return {'type': self.type, 'operand': self.operand.to_dict()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> UnaryMinus:
        """Reconstruct a UnaryMinus node from a dictionary.

        Args:
            data: Dictionary containing 'operand'.

        Returns:
            UnaryMinus instance.

        Raises:
            KeyError: If 'operand' is missing.

        """
        return UnaryMinus(operand=from_dict(data['operand']))


# ---------- Variables, historical access, functions ----------

@dataclass(frozen=True, slots=True)
class Let:
    """Variable binding (let expression).

    Attributes:
        var: Variable name.
        value: Expression to compute and assign.
        body: Expression where the variable is in scope.

    """

    value: ASTNode
    body: ASTNode
    type: str = 'Let'
    var: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Convert the Let node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'var', 'value', and 'body'.
            'value' and 'body' are serialized recursively.

        """
        return {
            'type': self.type,
            'var': self.var,
            'value': self.value.to_dict(),
            'body': self.body.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Let:
        """Reconstruct a Let node from a dictionary.

        Args:
            data: Dictionary containing 'var', 'value' and 'body'.

        Returns:
            Let instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Let(
            var=data['var'],
            value=from_dict(data['value']),
            body=from_dict(data['body']),
        )


@dataclass(frozen=True, slots=True)
class HistoricalAccess:
    """Historical (lagged) access to an indicator.
    Example: `close[1]`.

    Attributes:
        expr: Base indicator expression.
        offset: Number of bars back (0 = current).

    """

    expr: ASTNode
    type: str = 'HistoricalAccess'
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the HistoricalAccess node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'expr', and 'offset'.

        """
        return {
            'type': self.type,
            'expr': self.expr.to_dict(),
            'offset': self.offset
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> HistoricalAccess:
        """Reconstruct a HistoricalAccess node from a dictionary.

        Args:
            data: Dictionary containing 'expr' and 'offset'.

        Returns:
            HistoricalAccess instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return HistoricalAccess(
            expr=from_dict(data['expr']),
            offset=data['offset'],
        )


@dataclass(frozen=True, slots=True)
class Rising:
    """Rising function: true if expression has been strictly
    increasing over n bars.

    Attributes:
        expr: Indicator expression.
        n: Number of bars to check.

    """

    expr: ASTNode
    type: str = 'Rising'
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the Rising node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'expr', and 'n'.

        """
        return {'type': self.type, 'expr': self.expr.to_dict(), 'n': self.n}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Rising:
        """Reconstruct a Rising node from a dictionary.

        Args:
            data: Dictionary containing 'expr' and 'n'.

        Returns:
            Rising instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Rising(
            expr=from_dict(data['expr']),
            n=data['n'],
        )


@dataclass(frozen=True, slots=True)
class Falling:
    """Falling function: true if expression has been strictly
    decreasing over n bars.

    Attributes:
        expr: Indicator expression.
        n: Number of bars to check.

    """

    expr: ASTNode
    type: str = 'Falling'
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the Falling node to a JSON-serializable dictionary.

        Returns:
            A dict with keys 'type', 'expr', and 'n'.

        """
        return {'type': self.type, 'expr': self.expr.to_dict(), 'n': self.n}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Falling:
        """Reconstruct a Falling node from a dictionary.

        Args:
            data: Dictionary containing 'expr' and 'n'.

        Returns:
            Falling instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return Falling(
            expr=from_dict(data['expr']),
            n=data['n'],
        )


# ---------- AST type alias ----------
ASTNode = (
    Number | IndicatorAccess | IndicatorWithParams
    | Comparison | MultiComparison | LogicalBinOp | LogicalNot
    | Let | HistoricalAccess | Rising | Falling
    | Add | Sub | Mul | Div | Mod | Pow | UnaryMinus
)


# ---------- Deserialization factory ----------
def from_dict(data: dict[str, Any]) -> ASTNode:
    """Reconstruct an AST node from a JSON-serializable dictionary.

    This function examines the 'type' field in the input dict and delegates
    to the appropriate node's from_dict() method.

    Args:
        data: Dictionary containing at least a 'type' key.

    Returns:
        An instance of the appropriate AST node subclass.

    Raises:
        ValueError: If the 'type' value is unknown or the
        dictionary is invalid.

    """
    match data.get('type'):
        case 'Number':
            return Number.from_dict(data)
        case 'IndicatorAccess':
            return IndicatorAccess.from_dict(data)
        case 'IndicatorWithParams':
            return IndicatorWithParams.from_dict(data)
        case 'LogicalBinOp':
            return LogicalBinOp.from_dict(data)
        case 'LogicalNot':
            return LogicalNot.from_dict(data)
        case 'Comparison':
            return Comparison.from_dict(data)
        case 'MultiComparison':
            return MultiComparison.from_dict(data)
        case 'Add':
            return Add.from_dict(data)
        case 'Sub':
            return Sub.from_dict(data)
        case 'Mul':
            return Mul.from_dict(data)
        case 'Div':
            return Div.from_dict(data)
        case 'Mod':
            return Mod.from_dict(data)
        case 'Pow':
            return Pow.from_dict(data)
        case 'UnaryMinus':
            return UnaryMinus.from_dict(data)
        case 'Let':
            return Let.from_dict(data)
        case 'HistoricalAccess':
            return HistoricalAccess.from_dict(data)
        case 'Rising':
            return Rising.from_dict(data)
        case 'Falling':
            return Falling.from_dict(data)
        case _:
            raise ValueError(f'Unknown AST node type: {data.get("type")}')
