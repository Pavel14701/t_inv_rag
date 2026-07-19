"""AST interpreter with support for all operators."""

from typing import Any
from .ast import (
    ASTNode, Number, IndicatorAccess, IndicatorWithParams,
    Comparison, MultiComparison,
    LogicalBinOp, LogicalNot,
    Let, HistoricalAccess, Rising, Falling,
    Add, Sub, Mul, Div, Mod, Pow, UnaryMinus,
    Var,
)
from .context import Context
from .exceptions import EvaluationError


class Interpreter:
    """Interpreter for DSL AST.

    This class traverses the AST and evaluates it to a boolean result.
    It uses a context to obtain indicator values and supports:
    - Arithmetic operations (+, -, *, /, %, ^)
    - Comparisons (<, >, <=, >=, ==, !=)
    - Logical operations (and, or, not)
    - Indicator access with parameters
    - Historical access with bar offsets
    - Rising/falling functions
    - Let bindings with local variables

    Attributes:
        context: The context providing indicator values and validation.

    """

    def __init__(self, context: Context) -> None:
        """Initialize the interpreter with a context.

        Args:
            context: Context instance for indicator resolution.

        """
        self.context = context
        self._locals: dict[str, Any] = {}

    def visit(self, node: ASTNode) -> bool:
        """Visit an AST node and compute its boolean value.

        Args:
            node: The AST node to evaluate.

        Returns:
            The boolean result of the node evaluation.

        Raises:
            EvaluationError: If the node type is unknown or evaluation fails.

        """
        match node:
            case Number():
                return self._visit_number(node)
            case Var():
                return self._visit_var(node)
            case IndicatorAccess():
                return self._visit_indicator_access(node)
            case IndicatorWithParams():
                return self._visit_indicator_with_params(node)
            case Comparison():
                return self._visit_comparison(node)
            case MultiComparison():
                return self._visit_multi_comparison(node)
            case LogicalBinOp():
                return self._visit_logical_binop(node)
            case LogicalNot():
                return self._visit_logical_not(node)
            case Let():
                return self._visit_let(node)
            case HistoricalAccess():
                return self._visit_historical(node)
            case Rising():
                return self._visit_rising(node)
            case Falling():
                return self._visit_falling(node)
            case Add() | Sub() | Mul() | Div() | Mod() | Pow() | UnaryMinus():
                val = self._eval_arithmetic_node(node)
                return val != 0.0
            case _:
                raise EvaluationError(f'Unknown AST node: {type(node)}')

    # ---------- Arithmetic evaluation ----------

    def _eval_arithmetic_node(self, node: ASTNode) -> float:
        """Evaluate any AST node as a numeric value.

        Args:
            node: The AST node to evaluate as a number.

        Returns:
            The numeric value of the node.

        Raises:
            EvaluationError: If the node cannot be converted to a number
                or if division/modulo by zero occurs.

        """
        match node:
            case Number(value=val):
                return val
            case Var(name=var_name):
                return self._get_local_as_number(var_name)
            case IndicatorAccess(indicator=ind, attributes=attrs):
                if ind in self._locals:
                    return self._get_local_as_number(ind)
                return self._get_indicator_value(ind, {}, attrs, 0)
            case IndicatorWithParams(
                indicator=ind, params=params, attributes=attrs
            ):
                eval_params = self._eval_params(params)
                return self._get_indicator_value(ind, eval_params, attrs, 0)
            case HistoricalAccess():
                return self._visit_historical_as_number(node)
            case Add(left=left, right=right):
                return (
                    self._eval_arithmetic_node(left)
                    + self._eval_arithmetic_node(right)
                )
            case Sub(left=left, right=right):
                return (
                    self._eval_arithmetic_node(left)
                    - self._eval_arithmetic_node(right)
                )
            case Mul(left=left, right=right):
                return (
                    self._eval_arithmetic_node(left)
                    * self._eval_arithmetic_node(right)
                )
            case Div(left=left, right=right):
                right_val = self._eval_arithmetic_node(right)
                if right_val == 0:
                    raise EvaluationError('Division by zero')
                return self._eval_arithmetic_node(left) / right_val
            case Mod(left=left, right=right):
                right_val = self._eval_arithmetic_node(right)
                if right_val == 0:
                    raise EvaluationError('Modulo by zero')
                return self._eval_arithmetic_node(left) % right_val
            case Pow(left=left, right=right):
                return (
                    self._eval_arithmetic_node(left)
                    ** self._eval_arithmetic_node(right)
                )
            case UnaryMinus(operand=operand):
                return -self._eval_arithmetic_node(operand)
            case _:
                raise EvaluationError(
                    f'Cannot evaluate {type(node)} as number'
                )

    def _get_indicator_value(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Retrieve an indicator value from the context.

        Args:
            indicator: Indicator name.
            params: Parameter dictionary.
            attributes: List of attribute names.
            offset: Bar offset.

        Returns:
            The indicator value.

        Raises:
            EvaluationError: If the context cannot retrieve the value.

        """
        try:
            return self.context.get_value(
                indicator, params, attributes, offset
            )
        except Exception as e:
            raise EvaluationError(
                f'Indicator error for {indicator}: {e}'
            ) from e

    def _eval_params(self, params: dict[str, ASTNode]) -> dict[str, Any]:
        """Evaluate all parameter expressions to numeric values.

        Args:
            params: Dictionary of parameter names to AST nodes.

        Returns:
            Dictionary of parameter names to evaluated numeric values.

        """
        return {
            key: self._eval_arithmetic_node(node) for key,
            node in params.items()
        }

    def _get_local_as_number(self, name: str) -> float:
        """Resolve a local variable as a numeric value."""
        if name not in self._locals:
            raise EvaluationError(f'Undefined variable: {name}')
        value = self._locals[name]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        raise EvaluationError(
            f'Variable {name} is not a number (type {type(value)})'
        )

    # ---------- Visit methods for each node type ----------

    def _visit_number(self, node: Number) -> bool:
        """Convert a number to boolean (non-zero is True)."""
        return node.value != 0.0

    def _visit_var(self, node: Var) -> bool:
        """Evaluate a variable lookup."""
        if node.name not in self._locals:
            raise EvaluationError(f'Undefined variable: {node.name}')
        value = self._locals[node.name]
        return value if isinstance(value, bool) else float(value) != 0.0

    def _visit_indicator_access(self, node: IndicatorAccess) -> bool:
        """Evaluate an indicator access without parameters."""
        if node.indicator in self._locals:
            value = self._locals[node.indicator]
            return value if isinstance(value, bool) else float(value) != 0.0
        val = self._get_indicator_value(node.indicator, {}, node.attributes, 0)
        return val != 0.0

    def _visit_indicator_with_params(self, node: IndicatorWithParams) -> bool:
        """Evaluate an indicator access with parameters."""
        params = self._eval_params(node.params)
        val = self._get_indicator_value(
            node.indicator, params, node.attributes, 0
        )
        return val != 0.0

    def _visit_comparison(self, node: Comparison) -> bool:
        """Evaluate a binary comparison."""
        left_val = self._eval_arithmetic_node(node.left)
        right_val = self._eval_arithmetic_node(node.right)
        op = node.operator
        if op == '<':
            return left_val < right_val
        elif op == '>':
            return left_val > right_val
        elif op == '<=':
            return left_val <= right_val
        elif op == '>=':
            return left_val >= right_val
        elif op == '==':
            return left_val == right_val
        elif op == '!=':
            return left_val != right_val
        else:
            raise EvaluationError(f'Unknown comparison operator: {op}')

    def _visit_multi_comparison(self, node: MultiComparison) -> bool:
        """Evaluate a chained comparison (e.g., a < b <= c).

        All comparisons must hold sequentially.

        Args:
            node: The MultiComparison node.

        Returns:
            True if all comparisons hold, False otherwise.

        Raises:
            EvaluationError: If an unknown operator is encountered.

        """
        for i in range(len(node.operators)):
            left_val = self._eval_arithmetic_node(node.operands[i])
            right_val = self._eval_arithmetic_node(node.operands[i + 1])
            op = node.operators[i]
            match op:
                case '<':
                    if not (left_val < right_val):
                        return False
                case '>':
                    if not (left_val > right_val):
                        return False
                case '<=':
                    if not (left_val <= right_val):
                        return False
                case '>=':
                    if not (left_val >= right_val):
                        return False
                case '==':
                    if left_val != right_val:
                        return False
                case '!=':
                    if left_val == right_val:
                        return False
                case _:
                    raise EvaluationError(
                        f'Unknown comparison operator in chain: {op}'
                    )
        return True

    def _visit_logical_binop(self, node: LogicalBinOp) -> bool:
        """Evaluate a logical AND or OR operation (short-circuit)."""
        left = self.visit(node.left)
        if node.operator == 'and':
            return left and self.visit(node.right)
        elif node.operator == 'or':
            return left or self.visit(node.right)
        else:
            raise EvaluationError(f'Unknown logical operator: {node.operator}')

    def _visit_logical_not(self, node: LogicalNot) -> bool:
        """Evaluate a logical NOT operation."""
        return not self.visit(node.operand)

    def _visit_let(self, node: Let) -> bool:
        """Evaluate a let binding.

        The variable is bound to the computed value (numeric or boolean)
        and then the body is evaluated with the variable in scope.
        """
        # Compute the bound value: try numeric, then boolean
        try:
            bound = self._eval_arithmetic_node(node.value)
        except EvaluationError:
            bound = self.visit(node.value)  # boolean result

        # Enter new scope: save old locals, extend with new binding
        old_locals = self._locals
        new_locals = old_locals.copy()
        new_locals[node.var] = bound
        self._locals = new_locals

        try:
            result = self.visit(node.body)
        finally:
            self._locals = old_locals
        return result

    def _visit_historical(self, node: HistoricalAccess) -> bool:
        """Evaluate a historical access as boolean."""
        val = self._visit_historical_as_number(node)
        return val != 0.0

    def _visit_historical_as_number(self, node: HistoricalAccess) -> float:
        """Evaluate a historical access as a number.

        The expression must be an indicator access with or without parameters.
        """
        expr = node.expr
        if isinstance(expr, IndicatorAccess):
            return self._get_indicator_value(
                expr.indicator, {}, expr.attributes, node.offset
            )
        elif isinstance(expr, IndicatorWithParams):
            params = self._eval_params(expr.params)
            return self._get_indicator_value(
                expr.indicator, params, expr.attributes, node.offset
            )
        else:
            raise EvaluationError(
                f'HistoricalAccess expects indicator, got {type(expr)}'
            )

    def _visit_rising(self, node: Rising) -> bool:
        """Check if an indicator has strictly
        increased over the last n bars.
        """
        expr = node.expr
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('rising() expects an indicator expression')
        indicator = expr.indicator
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = self._eval_params(expr.params)
        attributes = expr.attributes
        history = self.context.get_history(
            indicator, params, attributes, node.n
        )
        if len(history) < node.n:
            return False
        return all(history[i] > history[i - 1] for i in range(1, len(history)))

    def _visit_falling(self, node: Falling) -> bool:
        """Check if an indicator has strictly
        decreased over the last n bars.
        """
        expr = node.expr
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('falling() expects an indicator expression')
        indicator = expr.indicator
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = self._eval_params(expr.params)
        attributes = expr.attributes
        history = self.context.get_history(
            indicator, params, attributes, node.n
        )
        if len(history) < node.n:
            return False
        return all(history[i] < history[i - 1] for i in range(1, len(history)))
