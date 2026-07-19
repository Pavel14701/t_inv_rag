"""Main entry point for evaluating DSL expressions."""

from .parser import parse
from .interpreter import Interpreter
from .context import Context


def evaluate_dsl(code: str, context: Context) -> bool:
    """Parse and evaluate a DSL expression with the given context.

    This function orchestrates the entire evaluation pipeline:
    1. Parses the input string into an AST.
    2. Creates an interpreter with the provided context.
    3. Visits the AST and returns the resulting boolean value.

    Args:
        code: The DSL expression string to evaluate.
        context: A Context instance providing indicator values and validation.

    Returns:
        The boolean result of the expression.

    Raises:
        ParseError: If the expression contains syntax errors.
        EvaluationError: If evaluation fails (e.g., unknown indicator,
        invalid parameters).

    """
    ast = parse(code)
    interpreter = Interpreter(context)
    return interpreter.visit(ast)
