"""AST interpreter with support for both synchronous and
asynchronous providers.

This interpreter evaluates the DSL AST to a boolean result. It provides two
evaluation modes: synchronous (`visit`) and asynchronous (`visit_async`). Both
use the same context to resolve indicator values, but asynchronous mode
supports non-blocking I/O when providers implement async interfaces.

The interpreter handles all language constructs: literals, variables,
indicator access with parameters and attributes, arithmetic operations,
comparisons (including chained), logical operations, let-bindings,
historical access, and rising/falling functions.
"""

from typing import Any

from .ast import (
    ASTNode,
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
)
from .context import Context
from .exceptions import EvaluationError


class Interpreter:
    """Interpreter for DSL AST with hybrid synchronous/asynchronous execution.

    This class traverses the AST and evaluates it to a boolean result.
    It uses the provided context to obtain indicator values. The context
    may support both synchronous and asynchronous provider resolution, allowing
    the same interpreter to be used in sync or async environments.

    Attributes:
        context: The context providing indicator values and validation.
        _locals: Dictionary of local variables for `let` bindings (shared
            across all visits; each visit updates it temporarily).

    Example:
        >>> ctx = Context([...])
        >>> interp = Interpreter(ctx)
        >>> ast = parse("close > 100")
        >>> result = interp.visit(ast)          # synchronous
        >>> result_async = await interp.visit_async(ast)  # asynchronous

    """

    def __init__(self, context: Context) -> None:
        """Initialize the interpreter with a context.

        Args:
            context: Context instance for indicator resolution.

        """
        self.context = context
        self._locals: dict[str, Any] = {}

    # ---------- Synchronous evaluation ----------

    def visit(self, node: ASTNode) -> bool:
        """Synchronously evaluate an AST node to a boolean value.

        This method traverses the AST recursively, evaluating each node
        according to its type. It uses the synchronous methods of the context
        (`get_value`, `get_history`) to resolve indicators.

        Args:
            node: The AST node to evaluate.

        Returns:
            The boolean result of the node evaluation.

        Raises:
            EvaluationError: If evaluation fails, e.g., division by zero,
                unknown indicator, invalid attribute, or type mismatch.

        Example:
            >>> interp = Interpreter(ctx)
            >>> ast = parse("rsi(period=14).value > 70")
            >>> interp.visit(ast)
            True

        """
        return self._visit_sync(node)

    def _visit_sync(self, node: ASTNode) -> bool:
        # sourcery skip: extract-method
        """Internal synchronous visitor (recursive).

        This method contains the main dispatch logic for all AST node types.
        It is called recursively during synchronous evaluation.

        Args:
            node: The AST node to evaluate.

        Returns:
            Boolean result of the node.

        Raises:
            EvaluationError: On invalid operations (division by zero, etc.)
                or unknown node types.

        """
        match node:
            case Number(value=val):
                return val != 0.0
            case Var(name=name):
                return self._get_local_as_bool(name)
            case IndicatorAccess(indicator=ind, attributes=attrs):
                if ind in self._locals:
                    return self._get_local_as_bool(ind)
                val = self.context.get_value(ind, {}, attrs, 0)
                return val != 0.0
            case IndicatorWithParams(
                indicator=ind,
                params=params,
                attributes=attrs
            ):
                eval_params = self._eval_params_sync(params)
                val = self.context.get_value(ind, eval_params, attrs, 0)
                return val != 0.0
            case Comparison(operator=op, left=left, right=right):
                left_val = self._eval_arith_sync(left)
                right_val = self._eval_arith_sync(right)
                return self._compare(op, left_val, right_val)
            case MultiComparison(operators=ops, operands=operands):
                return self._eval_multi_comp_sync(ops, operands)
            case LogicalBinOp(operator=op, left=left, right=right):
                left_bool = self._visit_sync(left)
                if op == 'and':
                    return left_bool and self._visit_sync(right)
                elif op == 'or':
                    return left_bool or self._visit_sync(right)
                else:
                    raise EvaluationError(f'Unknown logical operator: {op}')
            case LogicalNot(operand=operand):
                return not self._visit_sync(operand)
            case Let(var=name, value=value, body=body):
                # Compute the bound value: try numeric, then boolean
                try:
                    bound = self._eval_arith_sync(value)
                except EvaluationError:
                    bound = self._visit_sync(value)
                # Enter new scope
                old_locals = self._locals
                new_locals = old_locals.copy()
                new_locals[name] = bound
                self._locals = new_locals
                try:
                    result = self._visit_sync(body)
                finally:
                    self._locals = old_locals
                return result
            case HistoricalAccess(expr=expr, offset=offset):
                val = self._visit_historical_sync(expr, offset)
                return val != 0.0
            case Rising(expr=expr, n=n):
                return self._visit_rising_sync(expr, n)
            case Falling(expr=expr, n=n):
                return self._visit_falling_sync(expr, n)
            case Add(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    + self._eval_arith_sync(right)
                    != 0.0
                )
            case Sub(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    - self._eval_arith_sync(right)
                    != 0.0
                )
            case Mul(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    * self._eval_arith_sync(right)
                    != 0.0
                )
            case Div(left=left, right=right):
                right_val = self._eval_arith_sync(right)
                if right_val == 0:
                    raise EvaluationError('Division by zero')
                return self._eval_arith_sync(left) / right_val != 0.0
            case Mod(left=left, right=right):
                right_val = self._eval_arith_sync(right)
                if right_val == 0:
                    raise EvaluationError('Modulo by zero')
                return self._eval_arith_sync(left) % right_val != 0.0
            case Pow(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    ** self._eval_arith_sync(right)
                    != 0.0
                )
            case UnaryMinus(operand=operand):
                return -self._eval_arith_sync(operand) != 0.0
            case _:
                raise EvaluationError(f'Unknown AST node: {type(node)}')

    # ---------- Asynchronous evaluation ----------

    async def visit_async(self, node: ASTNode) -> bool:
        """Asynchronously evaluate an AST node to a boolean value.

        This method uses the asynchronous methods of the context
        (`get_value_async`, `get_history_async`) to resolve indicators,
        allowing non-blocking I/O when providers implement async interfaces.
        The traversal logic is identical to the synchronous version.

        Args:
            node: The AST node to evaluate.

        Returns:
            The boolean result of the node evaluation.

        Raises:
            EvaluationError: If evaluation fails (division by zero, unknown
                indicator, etc.).

        Example:
            >>> interp = Interpreter(ctx)
            >>> ast = parse("close > 100")
            >>> result = await interp.visit_async(ast)
            True

        """
        return await self._visit_async(node)

    async def _visit_async(self, node: ASTNode) -> bool:
        """Internal asynchronous visitor (recursive).

        This method mirrors `_visit_sync` but uses async calls for indicator
        resolution. It is called recursively during asynchronous evaluation.

        Args:
            node: The AST node to evaluate.

        Returns:
            Boolean result of the node.

        Raises:
            EvaluationError: On invalid operations or unknown node types.

        """
        match node:
            case Number(value=val):
                return val != 0.0
            case Var(name=name):
                return self._get_local_as_bool(name)
            case IndicatorAccess(indicator=ind, attributes=attrs):
                if ind in self._locals:
                    return self._get_local_as_bool(ind)
                val = await self.context.get_value_async(ind, {}, attrs, 0)
                return val != 0.0
            case IndicatorWithParams(
                indicator=ind,
                params=params,
                attributes=attrs
            ):
                eval_params = await self._eval_params_async(params)
                val = await self.context.get_value_async(
                    ind, eval_params, attrs, 0
                )
                return val != 0.0
            case Comparison(operator=op, left=left, right=right):
                left_val = await self._eval_arith_async(left)
                right_val = await self._eval_arith_async(right)
                return self._compare(op, left_val, right_val)
            case MultiComparison(operators=ops, operands=operands):
                return await self._eval_multi_comp_async(ops, operands)
            case LogicalBinOp(operator=op, left=left, right=right):
                left_bool = await self._visit_async(left)
                if op == 'and':
                    return left_bool and await self._visit_async(right)
                elif op == 'or':
                    return left_bool or await self._visit_async(right)
                else:
                    raise EvaluationError(f'Unknown logical operator: {op}')
            case LogicalNot(operand=operand):
                return not await self._visit_async(operand)
            case Let(var=name, value=value, body=body):
                # Try numeric, fallback to boolean
                try:
                    bound = await self._eval_arith_async(value)
                except EvaluationError:
                    bound = await self._visit_async(value)
                old_locals = self._locals
                new_locals = old_locals.copy()
                new_locals[name] = bound
                self._locals = new_locals
                try:
                    result = await self._visit_async(body)
                finally:
                    self._locals = old_locals
                return result
            case HistoricalAccess(expr=expr, offset=offset):
                val = await self._visit_historical_async(expr, offset)
                return val != 0.0
            case Rising(expr=expr, n=n):
                return await self._visit_rising_async(expr, n)
            case Falling(expr=expr, n=n):
                return await self._visit_falling_async(expr, n)
            case Add(left=left, right=right):
                return (
                    await self._eval_arith_async(left)
                    + await self._eval_arith_async(right)
                ) != 0.0
            case Sub(left=left, right=right):
                return (
                    await self._eval_arith_async(left)
                    - await self._eval_arith_async(right)
                ) != 0.0
            case Mul(left=left, right=right):
                return (
                    await self._eval_arith_async(left)
                    * await self._eval_arith_async(right)
                ) != 0.0
            case Div(left=left, right=right):
                right_val = await self._eval_arith_async(right)
                if right_val == 0:
                    raise EvaluationError('Division by zero')
                return (await self._eval_arith_async(left) / right_val) != 0.0
            case Mod(left=left, right=right):
                right_val = await self._eval_arith_async(right)
                if right_val == 0:
                    raise EvaluationError('Modulo by zero')
                return (await self._eval_arith_async(left) % right_val) != 0.0
            case Pow(left=left, right=right):
                return (
                    (await self._eval_arith_async(left))
                    ** (await self._eval_arith_async(right))
                ) != 0.0
            case UnaryMinus(operand=operand):
                return -(await self._eval_arith_async(operand)) != 0.0
            case _:
                raise EvaluationError(f'Unknown AST node: {type(node)}')

    # ---------- Arithmetic evaluation (synchronous) ----------

    def _eval_arith_sync(self, node: ASTNode) -> float:
        """Evaluate any AST node as a numeric value (synchronous).

        This method is used to compute the numeric value of an expression
        that appears in a context where a number is required (e.g., arithmetic
        operations, comparison operands). It handles numbers, variables,
        indicator accesses, historical accesses, and arithmetic operations.

        Args:
            node: The AST node to evaluate as a number.

        Returns:
            The numeric value of the expression.

        Raises:
            EvaluationError: If the node cannot be evaluated as a number
                (e.g., a boolean expression where a number is expected),
                or if division/modulo by zero occurs.

        """
        match node:
            case Number(value=val):
                return val
            case Var(name=name):
                return self._get_local_as_number(name)
            case IndicatorAccess(indicator=ind, attributes=attrs):
                if ind in self._locals:
                    return self._get_local_as_number(ind)
                return self.context.get_value(ind, {}, attrs, 0)
            case IndicatorWithParams(
                indicator=ind,
                params=params,
                attributes=attrs
            ):
                eval_params = self._eval_params_sync(params)
                return self.context.get_value(ind, eval_params, attrs, 0)
            case HistoricalAccess(expr=expr, offset=offset):
                return self._visit_historical_sync(expr, offset)
            case Add(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    + self._eval_arith_sync(right)
                )
            case Sub(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    - self._eval_arith_sync(right)
                )
            case Mul(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    * self._eval_arith_sync(right)
                )
            case Div(left=left, right=right):
                right_val = self._eval_arith_sync(right)
                if right_val == 0:
                    raise EvaluationError('Division by zero')
                return self._eval_arith_sync(left) / right_val
            case Mod(left=left, right=right):
                right_val = self._eval_arith_sync(right)
                if right_val == 0:
                    raise EvaluationError('Modulo by zero')
                return self._eval_arith_sync(left) % right_val
            case Pow(left=left, right=right):
                return (
                    self._eval_arith_sync(left)
                    ** self._eval_arith_sync(right)
                )
            case UnaryMinus(operand=operand):
                return -self._eval_arith_sync(operand)
            case _:
                raise EvaluationError(
                    f'Cannot evaluate {type(node)} as number'
                )

    def _eval_params_sync(self, params: dict[str, ASTNode]) -> dict[str, Any]:
        """Evaluate all parameter expressions to numeric values synchronously.

        This method is used to process the parameter dictionary of an indicator
        call, converting each AST expression into a concrete numeric value.

        Args:
            params: Dictionary mapping parameter names to AST nodes.

        Returns:
            Dictionary with the same keys but with evaluated numeric values.

        """
        return {
            key: self._eval_arith_sync(node)
            for key, node in params.items()
        }

    # ---------- Arithmetic evaluation (asynchronous) ----------

    async def _eval_arith_async(self, node: ASTNode) -> float:
        """Evaluate any AST node as a numeric value (asynchronous).

        This method mirrors `_eval_arith_sync` but uses asynchronous methods
        of the context to resolve indicator values.

        Args:
            node: The AST node to evaluate as a number.

        Returns:
            The numeric value.

        Raises:
            EvaluationError: If the node cannot be evaluated as a number,
                or division/modulo by zero occurs.

        """
        match node:
            case Number(value=val):
                return val
            case Var(name=name):
                return self._get_local_as_number(name)
            case IndicatorAccess(indicator=ind, attributes=attrs):
                if ind in self._locals:
                    return self._get_local_as_number(ind)
                return await self.context.get_value_async(ind, {}, attrs, 0)
            case IndicatorWithParams(
                indicator=ind,
                params=params,
                attributes=attrs
            ):
                eval_params = await self._eval_params_async(params)
                return await self.context.get_value_async(
                    ind, eval_params, attrs, 0
                )
            case HistoricalAccess(expr=expr, offset=offset):
                return await self._visit_historical_async(expr, offset)
            case Add(left=left, right=right):
                return (await self._eval_arith_async(left)
                        + await self._eval_arith_async(right))
            case Sub(left=left, right=right):
                return (await self._eval_arith_async(left)
                        - await self._eval_arith_async(right))
            case Mul(left=left, right=right):
                return (await self._eval_arith_async(left)
                        * await self._eval_arith_async(right))
            case Div(left=left, right=right):
                right_val = await self._eval_arith_async(right)
                if right_val == 0:
                    raise EvaluationError('Division by zero')
                return (await self._eval_arith_async(left)) / right_val
            case Mod(left=left, right=right):
                right_val = await self._eval_arith_async(right)
                if right_val == 0:
                    raise EvaluationError('Modulo by zero')
                return (await self._eval_arith_async(left)) % right_val
            case Pow(left=left, right=right):
                return ((await self._eval_arith_async(left))
                        ** (await self._eval_arith_async(right)))
            case UnaryMinus(operand=operand):
                return -(await self._eval_arith_async(operand))
            case _:
                raise EvaluationError(
                    f'Cannot evaluate {type(node)} as number'
                )

    async def _eval_params_async(
        self,
        params: dict[str, ASTNode]
    ) -> dict[str, Any]:
        """Evaluate all parameter expressions to numeric values asynchronously.

        Args:
            params: Dictionary mapping parameter names to AST nodes.

        Returns:
            Dictionary with evaluated numeric values.

        """
        result = {}
        for key, node in params.items():
            result[key] = await self._eval_arith_async(node)
        return result

    # ---------- Local variable utilities ----------

    def _get_local_as_bool(self, name: str) -> bool:
        """Retrieve a local variable's value as a boolean.

        This method handles conversion of numeric values to booleans
        (non-zero → True, zero → False). If the variable is already boolean,
        it returns it as-is.

        Args:
            name: Name of the variable.

        Returns:
            Boolean value of the variable.

        Raises:
            EvaluationError: If the variable is not defined or its type is
                neither boolean nor numeric.

        """
        if name not in self._locals:
            raise EvaluationError(f'Undefined variable: {name}')
        value = self._locals[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0.0
        raise EvaluationError(
            f'Variable {name} is not boolean or numeric: {type(value)}'
        )

    def _get_local_as_number(self, name: str) -> float:
        """Retrieve a local variable's value as a number.

        Booleans are converted to 1.0 (True) or 0.0 (False).

        Args:
            name: Name of the variable.

        Returns:
            Numeric value of the variable.

        Raises:
            EvaluationError: If the variable is not defined or its type is
                neither numeric nor boolean.

        """
        if name not in self._locals:
            raise EvaluationError(f'Undefined variable: {name}')
        value = self._locals[name]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        raise EvaluationError(f'Variable {name} is not numeric: {type(value)}')

    # ---------- Comparison utilities ----------

    def _compare(self, op: str, left: float, right: float) -> bool:
        """Perform a single comparison operation.

        Args:
            op: Comparison operator ('<', '>', '<=', '>=', '==', '!=').
            left: Left operand value.
            right: Right operand value.

        Returns:
            Boolean result of the comparison.

        Raises:
            EvaluationError: If the operator is unknown.

        """
        if op == '<':
            return left < right
        elif op == '>':
            return left > right
        elif op == '<=':
            return left <= right
        elif op == '>=':
            return left >= right
        elif op == '==':
            return left == right
        elif op == '!=':
            return left != right
        else:
            raise EvaluationError(f'Unknown comparison operator: {op}')

    def _eval_multi_comp_sync(
        self,
        ops: list[str],
        operands: list[ASTNode]
    ) -> bool:
        """Evaluate a chained comparison synchronously (e.g., a < b <= c).

        Each comparison in the chain is evaluated in sequence; if any fails,
        the whole chain returns False.

        Args:
            ops: List of comparison operators.
            operands: List of operand AST nodes (length = len(ops) + 1).

        Returns:
            True if all comparisons hold, False otherwise.

        """
        for i in range(len(ops)):
            left_val = self._eval_arith_sync(operands[i])
            right_val = self._eval_arith_sync(operands[i + 1])
            if not self._compare(ops[i], left_val, right_val):
                return False
        return True

    async def _eval_multi_comp_async(
        self,
        ops: list[str],
        operands: list[ASTNode]
    ) -> bool:
        """Evaluate a chained comparison asynchronously.

        Args:
            ops: List of comparison operators.
            operands: List of operand AST nodes.

        Returns:
            True if all comparisons hold, False otherwise.

        """
        for i in range(len(ops)):
            left_val = await self._eval_arith_async(operands[i])
            right_val = await self._eval_arith_async(operands[i + 1])
            if not self._compare(ops[i], left_val, right_val):
                return False
        return True

    # ---------- Historical access ----------

    def _visit_historical_sync(self, expr: ASTNode, offset: int) -> float:
        """Evaluate a historical access synchronously.

        The expression must be an indicator access
        (with or without parameters).
        The offset determines how many bars
        back to fetch.

        Args:
            expr: The base expression (IndicatorAccess or IndicatorWithParams).
            offset: Number of bars back (0 = current, 1 = previous bar, etc.).

        Returns:
            Numeric value of the historical indicator.

        Raises:
            EvaluationError: If expr is not an indicator access.

        """
        if isinstance(expr, IndicatorAccess):
            return self.context.get_value(
                expr.indicator,
                {},
                expr.attributes,
                offset
            )
        elif isinstance(expr, IndicatorWithParams):
            params = self._eval_params_sync(expr.params)
            return self.context.get_value(
                expr.indicator,
                params,
                expr.attributes,
                offset
            )
        else:
            raise EvaluationError(
                f'HistoricalAccess expects indicator, got {type(expr)}'
            )

    async def _visit_historical_async(
        self,
        expr: ASTNode,
        offset: int
    ) -> float:
        """Evaluate a historical access asynchronously.

        Args:
            expr: The base expression (IndicatorAccess or IndicatorWithParams).
            offset: Number of bars back.

        Returns:
            Numeric value of the historical indicator.

        Raises:
            EvaluationError: If expr is not an indicator access.

        """
        if isinstance(expr, IndicatorAccess):
            return await self.context.get_value_async(
                expr.indicator,
                {},
                expr.attributes,
                offset
            )
        elif isinstance(expr, IndicatorWithParams):
            params = await self._eval_params_async(expr.params)
            return await self.context.get_value_async(
                expr.indicator,
                params,
                expr.attributes,
                offset
            )
        else:
            raise EvaluationError(
                f'HistoricalAccess expects indicator, got {type(expr)}'
            )

    # ---------- Rising / Falling ----------

    def _visit_rising_sync(self, expr: ASTNode, n: int) -> bool:
        """Check if an indicator has strictly increased over the
        last n bars (sync).

        The function retrieves the last n historical values (including current)
        and verifies that each value is greater than the previous one.

        Args:
            expr: Indicator expression
                (IndicatorAccess or IndicatorWithParams).
            n: Number of bars to check (must be > 0).

        Returns:
            True if the indicator strictly increased over the last n bars,
            False otherwise (or if insufficient history is available).

        """
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('rising() expects an indicator expression')
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = self._eval_params_sync(expr.params)
        attrs = expr.attributes
        indicator = expr.indicator
        history = self.context.get_history(indicator, params, attrs, n)
        if len(history) < n:
            return False
        return all(history[i] > history[i - 1] for i in range(1, len(history)))

    async def _visit_rising_async(self, expr: ASTNode, n: int) -> bool:
        """Check if an indicator has strictly increased
        over the last n bars (async).

        Args:
            expr: Indicator expression.
            n: Number of bars to check.

        Returns:
            True if strictly increased, False otherwise.

        """
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('rising() expects an indicator expression')
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = await self._eval_params_async(expr.params)
        attrs = expr.attributes
        indicator = expr.indicator
        history = await self.context.get_history_async(
            indicator, params, attrs, n
        )
        if len(history) < n:
            return False
        return all(history[i] > history[i - 1] for i in range(1, len(history)))

    def _visit_falling_sync(self, expr: ASTNode, n: int) -> bool:
        """Check if an indicator has strictly decreased over the
        last n bars (sync).

        Args:
            expr: Indicator expression.
            n: Number of bars to check.

        Returns:
            True if strictly decreased, False otherwise.

        """
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('falling() expects an indicator expression')
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = self._eval_params_sync(expr.params)
        attrs = expr.attributes
        indicator = expr.indicator
        history = self.context.get_history(indicator, params, attrs, n)
        if len(history) < n:
            return False
        return all(history[i] < history[i - 1] for i in range(1, len(history)))

    async def _visit_falling_async(self, expr: ASTNode, n: int) -> bool:
        """Check if an indicator has strictly decreased over
        the last n bars (async).

        Args:
            expr: Indicator expression.
            n: Number of bars to check.

        Returns:
            True if strictly decreased, False otherwise.

        """
        if not isinstance(expr, (IndicatorAccess, IndicatorWithParams)):
            raise EvaluationError('falling() expects an indicator expression')
        if isinstance(expr, IndicatorAccess):
            params = {}
        else:
            params = await self._eval_params_async(expr.params)
        attrs = expr.attributes
        indicator = expr.indicator
        history = await self.context.get_history_async(
            indicator, params, attrs, n
        )
        if len(history) < n:
            return False
        return all(history[i] < history[i - 1] for i in range(1, len(history)))
