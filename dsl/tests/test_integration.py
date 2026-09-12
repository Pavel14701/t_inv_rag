"""Integration tests for the interpreter with providers.

This module tests the full pipeline: tokenization -> parsing -> interpretation
with actual context and providers.
"""

import pytest

from ..ast import ASTNode
from ..context import Context
from ..evaluate import evaluate_dsl, evaluate_dsl_async
from ..exceptions import EvaluationError, ProviderError
from ..interpreter import Interpreter
from ..parser import Parser
from .conftest import DEFAULT_MANIFEST


def parse(code: str) -> ASTNode:
    """Parse DSL code into AST."""
    p = Parser()
    return p.parse(code)


@pytest.mark.integration
@pytest.mark.without_providers
def test_literals_and_arithmetic_no_providers(context_empty) -> None:
    """Test that expressions without indicators work without any provider."""
    assert evaluate_dsl("1 + 2", context_empty) is True
    assert evaluate_dsl("5 - 5", context_empty) is False
    assert evaluate_dsl("2 * 0", context_empty) is False
    assert evaluate_dsl("10 / 2", context_empty) is True
    assert evaluate_dsl("2 ^ 3", context_empty) is True
    assert evaluate_dsl("(1 + 2) * 3", context_empty) is True
    with pytest.raises(EvaluationError, match="Division by zero"):
        evaluate_dsl("1 / 0", context_empty)


@pytest.mark.integration
@pytest.mark.without_providers
def test_comparisons_no_providers(context_empty) -> None:
    """Test comparisons without providers."""
    assert evaluate_dsl("5 > 3", context_empty) is True
    assert evaluate_dsl("5 < 3", context_empty) is False
    assert evaluate_dsl("5 == 5", context_empty) is True
    assert evaluate_dsl("5 != 5", context_empty) is False


@pytest.mark.integration
@pytest.mark.without_providers
def test_logical_no_providers(context_empty) -> None:
    """Test logical operations without providers."""
    assert evaluate_dsl("1 and 1", context_empty) is True
    assert evaluate_dsl("1 and 0", context_empty) is False
    assert evaluate_dsl("0 or 1", context_empty) is True
    assert evaluate_dsl("not 1", context_empty) is False


@pytest.mark.integration
@pytest.mark.without_providers
def test_let_no_providers(context_empty) -> None:
    """Test let binding without providers."""
    assert evaluate_dsl("let x = 5 in x > 3", context_empty) is True
    assert evaluate_dsl("let x = 5 + 3 in x * 2", context_empty) is True
    assert (
        evaluate_dsl("let x = 5 in let y = x + 1 in y == 6", context_empty)
        is True
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_simple_indicator_access(context_with_sample_data) -> None:
    """Test simple indicator access with provider."""
    assert evaluate_dsl("close", context_with_sample_data) is True
    assert evaluate_dsl("close > 50", context_with_sample_data) is True
    assert evaluate_dsl("close < 50", context_with_sample_data) is False


@pytest.mark.integration
@pytest.mark.with_providers
def test_indicator_with_params(context_with_sample_data) -> None:
    """Test indicator with parameters."""
    assert (
        evaluate_dsl("rsi(period=14).value > 70", context_with_sample_data)
        is True
    )
    assert (
        evaluate_dsl("rsi(period=14).value < 70", context_with_sample_data)
        is False
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_arithmetic_with_indicators(context_with_sample_data) -> None:
    """Test arithmetic combining indicators and numbers."""
    assert (
        evaluate_dsl("close * 2 + volume / 10 - 50", context_with_sample_data)
        is True
    )
    assert evaluate_dsl("close - 100", context_with_sample_data) is False


@pytest.mark.integration
@pytest.mark.with_providers
def test_chained_comparison_with_indicators(context_with_sample_data) -> None:
    """Test chained comparisons with indicators."""
    # low=50, close=100, high=110 are already in sample_values fixture
    assert (
        evaluate_dsl("low < close <= high", context_with_sample_data) is True
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_let_with_indicator(context_with_sample_data) -> None:
    """Test let binding with indicator values."""
    assert (
        evaluate_dsl("let x = close in x > 50", context_with_sample_data)
        is True
    )
    assert (
        evaluate_dsl("let x = close - 100 in x > 0", context_with_sample_data)
        is False
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_historical_access(context_with_sample_data) -> None:
    """Test historical access with provider."""
    assert evaluate_dsl("close[1] > 90", context_with_sample_data) is True
    assert evaluate_dsl("close[1] > 100", context_with_sample_data) is False


@pytest.mark.integration
@pytest.mark.with_providers
def test_rising_function(context_with_sample_data) -> None:
    """Test rising function with historical data."""
    assert evaluate_dsl("rising(close, 2)", context_with_sample_data) is True
    # close history: [10, 20, 30, 40] ->
    # rising over 2 bars: 20->30, 30->40 = True
    assert evaluate_dsl("rising(close, 4)", context_with_sample_data) is True
    # 10->20->30->40 = True


@pytest.mark.integration
@pytest.mark.with_providers
def test_falling_function(context_with_sample_data) -> None:
    """Test falling function with historical data."""
    # We need a decreasing history
    provider = context_with_sample_data.providers[0]
    provider.history["close", (), ()] = [40, 30, 20, 10]
    assert evaluate_dsl("falling(close, 2)", context_with_sample_data) is True
    assert evaluate_dsl("falling(close, 4)", context_with_sample_data) is True


@pytest.mark.integration
@pytest.mark.with_providers
def test_complex_expression_with_operators(context_with_sample_data) -> None:
    """Test complex expression mixing indicators,
    arithmetic, and comparisons.
    """
    assert (
        evaluate_dsl("(-close ^ 2) + 5 * close - 3", context_with_sample_data)
        is True
    )
    assert (
        evaluate_dsl("close > 0 and not (close < 5)", context_with_sample_data)
        is True
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_indicator_with_multiple_params(context_with_sample_data) -> None:
    """Test indicator with multiple parameters."""
    assert (
        evaluate_dsl(
            "macd(fast=12, slow=26).line > macd(fast=12, slow=26).signal",
            context_with_sample_data,
        )
        is True
    )


@pytest.mark.integration
@pytest.mark.with_providers
def test_multiple_providers_fallback() -> None:
    """Test that context tries providers in order, falling back if needed."""
    from dsl.providers.base import IndicatorProvider

    class FailingProvider(IndicatorProvider):
        def get_manifest(self) -> dict:
            return DEFAULT_MANIFEST

        def resolve(
            self, indicator: str, params: dict, attributes: list, offset: int
        ) -> float:
            raise ProviderError("Fail")

    class WorkingProvider(IndicatorProvider):
        def __init__(self) -> None:
            self.values = {("close", (), (), 0): 100.0}

        def get_manifest(self) -> dict:
            return DEFAULT_MANIFEST

        def resolve(
            self, indicator: str, params: dict, attributes: list, offset: int
        ) -> float:
            if indicator == "close":
                return self.values.get((indicator, (), (), offset), 0.0)
            raise ProviderError("Unknown")

    ctx = Context([FailingProvider(), WorkingProvider()])
    assert evaluate_dsl("close", ctx) is True
    assert evaluate_dsl("close == 100", ctx) is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_indicator_access(async_context_with_sample_data) -> None:
    """Test asynchronous indicator access."""
    ast = parse("close > 50")
    interp = Interpreter(async_context_with_sample_data)
    result = await interp.visit_async(ast)
    assert result is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_arithmetic_with_indicators(
    async_context_with_sample_data,
) -> None:
    """Test asynchronous arithmetic with indicators."""
    assert (
        await evaluate_dsl_async(
            "close * 2 + volume / 10 - 50", async_context_with_sample_data
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_historical_access(async_context_with_sample_data) -> None:
    """Test asynchronous historical access."""
    assert (
        await evaluate_dsl_async(
            "close[1] > 90", async_context_with_sample_data
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_rising(async_context_with_sample_data) -> None:
    """Test asynchronous rising function."""
    assert (
        await evaluate_dsl_async(
            "rising(close, 2)", async_context_with_sample_data
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_falling(async_context_with_sample_data) -> None:
    """Test asynchronous falling function."""
    provider = async_context_with_sample_data.providers[0]
    provider.history["close", (), ()] = [40, 30, 20, 10]
    assert (
        await evaluate_dsl_async(
            "falling(close, 3)", async_context_with_sample_data
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.async_test
async def test_async_error_handling(async_context_with_sample_data) -> None:
    """Test error handling in async mode."""
    with pytest.raises(EvaluationError, match="Division by zero"):
        await evaluate_dsl_async("1 / 0", async_context_with_sample_data)
    with pytest.raises(
        ValueError, match="Unknown indicator: unknown_indicator"
    ):
        await evaluate_dsl_async(
            "unknown_indicator", async_context_with_sample_data
        )
