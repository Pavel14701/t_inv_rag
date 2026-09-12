"""Unit tests for the evaluate module.

This module tests the high-level evaluate_dsl and evaluate_dsl_async functions.
"""

import pytest

from ..evaluate import evaluate_dsl, evaluate_dsl_async
from ..exceptions import EvaluationError, ParseError


@pytest.mark.unit
@pytest.mark.evaluate
def test_evaluate_simple(context_empty) -> None:
    """Test evaluate_dsl with simple expressions."""
    assert evaluate_dsl("1 + 2", context_empty) is True
    assert evaluate_dsl("5 - 5", context_empty) is False


@pytest.mark.unit
@pytest.mark.evaluate
def test_evaluate_with_indicators(context_with_sample_data) -> None:
    """Test evaluate_dsl with indicators."""
    assert evaluate_dsl("close > 50", context_with_sample_data) is True
    assert (
        evaluate_dsl("rsi(period=14).value > 70", context_with_sample_data)
        is True
    )


@pytest.mark.unit
@pytest.mark.evaluate
def test_evaluate_errors(context_empty) -> None:
    """Test evaluate_dsl error handling."""
    with pytest.raises(ParseError):
        evaluate_dsl("1 +", context_empty)
    with pytest.raises(EvaluationError, match="Division by zero"):
        evaluate_dsl("1 / 0", context_empty)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.evaluate
@pytest.mark.async_test
async def test_evaluate_async_simple(async_context_with_sample_data) -> None:
    """Test evaluate_dsl_async with simple expressions."""
    assert (
        await evaluate_dsl_async("1 + 2", async_context_with_sample_data)
        is True
    )
    assert (
        await evaluate_dsl_async("5 - 5", async_context_with_sample_data)
        is False
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.evaluate
@pytest.mark.async_test
async def test_evaluate_async_with_indicators(
    async_context_with_sample_data,
) -> None:
    """Test evaluate_dsl_async with indicators."""
    assert (
        await evaluate_dsl_async("close > 50", async_context_with_sample_data)
        is True
    )
    assert (
        await evaluate_dsl_async(
            "rsi(period=14).value > 70", async_context_with_sample_data
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.evaluate
@pytest.mark.async_test
async def test_evaluate_async_errors(async_context_with_sample_data) -> None:
    """Test evaluate_dsl_async error handling."""
    with pytest.raises(ParseError):
        await evaluate_dsl_async("1 +", async_context_with_sample_data)
    with pytest.raises(EvaluationError, match="Division by zero"):
        await evaluate_dsl_async("1 / 0", async_context_with_sample_data)
