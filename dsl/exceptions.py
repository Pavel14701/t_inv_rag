"""DSL exceptions hierarchy."""


class DSLError(Exception):
    """Base exception for all DSL-related errors."""


class ParseError(DSLError):
    """Raised when the input expression cannot be parsed due to syntax errors.

    This typically occurs when the expression does not conform
    to the DSL grammar. The error message should include the position
    of the error and the expected token.
    """


class EvaluationError(DSLError):
    """Raised when an expression fails during evaluation.

    This can happen for various reasons:
    - Unknown indicator name.
    - Invalid attribute path for an indicator.
    - Type mismatch in comparisons (e.g., comparing string to number).
    - Division by zero or other arithmetic errors.
    - Unbound variable in a `let` expression.
    """


class ProviderError(DSLError):
    """Raised when an indicator provider fails to resolve a value.

    This exception is intended to be raised by providers when they cannot
    compute the requested indicator value due to missing data, invalid
    parameters, or internal errors. The DSL context will catch
    this exception and attempt the next available provider.
    """


class DslValidationError(DSLError, ValueError):
    """Raised when an indicator request violates the manifest schema.

    Наследует и ``DSLError`` (единый контракт для RAG repair-loop —
    TZ-01 п.2.2), и ``ValueError`` (обратная совместимость: ранее
    ``Context._validate`` бросал голый ValueError, и существующие
    клиенты ловят его). Текст содержит человекочитаемые формулировки
    манифест-валидатора (``Unknown indicator: X``, ``Invalid parameter
    'Y' for indicator 'Z'``) — они парсятся RAG-ом как repair-подсказки.
    """
