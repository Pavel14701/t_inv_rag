"""Main entry point for evaluating DSL expressions."""

from .parser import parse
from .interpreter import Interpreter
from .context import Context


def evaluate_dsl(code: str, context: Context) -> bool:
    """Parse and evaluate a DSL expression synchronously.

    Args:
        code: The DSL expression string to evaluate.
        context: A Context instance providing indicator values and validation.

    Returns:
        The boolean result of the expression.

    Raises:
        ParseError: If the expression contains syntax errors.
        EvaluationError: If evaluation fails.

    """
    ast = parse(code)
    interpreter = Interpreter(context)
    return interpreter.visit(ast)


async def evaluate_dsl_async(code: str, context: Context) -> bool:
    """Parse and evaluate a DSL expression asynchronously.

    This function uses the asynchronous methods of the context to resolve
    indicators, allowing non-blocking I/O with async providers.

    Args:
        code: The DSL expression string to evaluate.
        context: A Context instance providing indicator values and validation.

    Returns:
        The boolean result of the expression.

    Raises:
        ParseError: If the expression contains syntax errors.
        EvaluationError: If evaluation fails.

    """
    ast = parse(code)
    interpreter = Interpreter(context)
    return await interpreter.visit_async(ast)
