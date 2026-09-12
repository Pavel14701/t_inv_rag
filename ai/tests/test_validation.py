"""Unit tests for the validation module.

This module tests the contract validation functions that verify the shape,
dtype, and value constraints of tensors and order blocks used in the
EntryExitTransformer pipeline.
"""

import pytest
import torch

from torch import Tensor

from ai.src.contracts import (
    validate_action_targets,
    validate_batch,
    validate_indicators,
    validate_order_blocks,
    validate_outcome_targets,
    validate_prices,
    validate_signals,
    validate_tp_sl,
)
from ai.src.datatypes import OrderBlock


@pytest.mark.unit
def test_validate_prices(sample_batch: tuple) -> None:
    """Test price tensor validation.

    The price tensor should have shape (B, T, n_price_feats), dtype float,
    finite values, and positive prices (only warning for non-positive).

    Args:
        sample_batch: Fixture with a sample batch (11-element tuple).

    Asserts:
        - Validation passes for a valid price tensor.
        - Validation raises ValueError for incorrect last dimension.
        - Validation raises ValueError for NaN/Inf values.
        - Non-positive prices trigger a warning (not an error).

    """
    prices: Tensor = sample_batch[0]  # (B, T, n_price_feats)
    n_price_feats: int = prices.shape[-1]
    # Should pass
    validate_prices(prices, n_price_feats)
    # Wrong last dim
    with pytest.raises(ValueError, match="last dim must be"):
        validate_prices(prices, n_price_feats + 1)
    # NaN
    prices_nan = prices.clone()
    prices_nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_prices(prices_nan, n_price_feats)
    # Non-positive (warns, not raises)
    prices_neg = prices.clone()
    prices_neg[0, 0, 0] = -1.0
    validate_prices(prices_neg, n_price_feats)  # should only print warning


@pytest.mark.unit
def test_validate_indicators(sample_batch: tuple) -> None:
    """Test indicator tensor validation.

    The indicator tensor should have shape (B, T, n_ind_feats), dtype float,
    finite values.  If n_ind_feats=0, the tensor should have zero features.

    Args:
        sample_batch: Fixture with a sample batch.

    Asserts:
        - Validation passes for a valid indicator tensor.
        - Validation raises ValueError for incorrect last dimension.
        - Zero features with non-zero tensor triggers a warning.

    """
    indicators: Tensor = sample_batch[1]  # (B, T, n_ind_feats)
    n_ind_feats: int = indicators.shape[-1]
    # Pass
    validate_indicators(indicators, n_ind_feats)
    # Wrong dim
    with pytest.raises(ValueError, match="last dim must be"):
        validate_indicators(indicators, n_ind_feats + 1)
    # Zero features but non-zero tensor (warn)
    validate_indicators(indicators, 0)  # should warn


@pytest.mark.unit
def test_validate_signals(sample_batch: tuple) -> None:
    """Test signal tensor validation.

    The signal tensor should have shape (B, T, n_sig_feats), dtype float,
    finite values.

    Args:
        sample_batch: Fixture with a sample batch.

    Asserts:
        - Validation passes for a valid signal tensor.
        - Validation raises ValueError for incorrect last dimension.

    """
    signals: Tensor = sample_batch[2]
    n_sig_feats: int = signals.shape[-1]
    validate_signals(signals, n_sig_feats)
    with pytest.raises(ValueError, match="last dim must be"):
        validate_signals(signals, n_sig_feats + 1)


@pytest.mark.unit
def test_validate_tp_sl() -> None:
    """Test TP and SL tensor validation.

    Both TP and SL should have shape (B, T, 1), dtype float, finite,
    and positive absolute prices.

    Asserts:
        - Validation passes for positive TP and SL.
        - Validation raises ValueError for negative TP/SL.

    """
    B, T = 2, 10  # noqa: N806
    tp: Tensor = torch.rand(B, T, 1) * 100 + 50  # from 50 to 150
    sl: Tensor = torch.rand(B, T, 1) * 50  # from 0 to 50
    validate_tp_sl(tp, sl)  # should pass
    # Negative values
    tp_neg: Tensor = -torch.abs(tp)
    with pytest.raises(ValueError, match="positive"):
        validate_tp_sl(tp_neg, sl)


@pytest.mark.unit
def test_validate_order_blocks(
    sample_batch: tuple,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test order block validation.

    The order block list should have length equal to batch size, each element
    a list of OrderBlock objects with valid indices and fields.

    Args:
        sample_batch: Fixture with a sample batch.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - Validation passes for a valid list of order blocks.
        - Validation raises ValueError for incorrect batch size.
        - Validation raises TypeError for invalid object types.

    """
    ob_list: list[list[OrderBlock]] = sample_batch[5]  # list of lists
    batch_size: int = len(ob_list)
    seq_len: int = sample_batch[0].shape[1]
    # Pass
    validate_order_blocks(ob_list, batch_size, seq_len)
    # Wrong batch size
    with pytest.raises(ValueError, match="batch size"):
        validate_order_blocks(ob_list, batch_size + 1, seq_len)
    # Invalid object (not a list)
    with pytest.raises(TypeError):
        validate_order_blocks([1, 2], batch_size, seq_len)  # type: ignore[list-item]


@pytest.mark.unit
def test_validate_action_targets(sample_batch: tuple) -> None:
    """Test action target validation.

    Action targets should have shape (B, T), dtype float (or int), and values
    in {-100, 0, 1, 2}.

    Args:
        sample_batch: Fixture with a sample batch.

    Asserts:
        - Validation passes for valid action targets.
        - Validation raises ValueError for invalid values.

    """
    # Convert action targets to float32 because validator expects float
    action_tgt: Tensor = sample_batch[6].float()
    # Valid values: -100, 0, 1, 2
    validate_action_targets(action_tgt)  # should pass
    # Invalid value
    action_invalid = action_tgt.clone()
    action_invalid[0, 0] = 5.0
    with pytest.raises(ValueError, match="invalid values"):
        validate_action_targets(action_invalid)


@pytest.mark.unit
def test_validate_outcome_targets(sample_batch: tuple) -> None:
    """Test outcome target validation.

    Outcome targets should have shape (B, T), dtype float, finite values.
    In binary/multiclass modes, finite values must be {0.0, 1.0, 2.0}.
    In regression mode, only a warning for very large values.

    Args:
        sample_batch: Fixture with a sample batch.

    Asserts:
        - Validation passes for valid outcome targets in binary mode.
        - Validation raises ValueError for invalid finite values in
            binary mode.
        - Regression mode only warns for large values.

    """
    outcome_tgt: Tensor = sample_batch[7]
    # Binary mode
    validate_outcome_targets(outcome_tgt, "binary")
    # Invalid finite values
    outcome_invalid = outcome_tgt.clone()
    outcome_invalid[0, 0] = 3.0
    with pytest.raises(ValueError, match="invalid finite values"):
        validate_outcome_targets(outcome_invalid, "binary")
    # Regression mode: should not raise, only warn
    outcome_big = outcome_tgt.clone()
    outcome_big[0, 0] = 1e7
    validate_outcome_targets(outcome_big, "regression")  # warns


@pytest.mark.integration
def test_validate_batch(sample_batch: tuple) -> None:
    """Test full batch validation.

    The batch should contain 11 elements, all tensors and lists
    correctly shaped. This test also ensures TP/SL positivity
    and action target dtype.

    Args:
        sample_batch: Fixture with a sample batch (11-element tuple).

    Asserts:
        - Validation passes for a valid batch.

    """
    batch = list(sample_batch)
    # Ensure TP and SL are positive
    tp: Tensor = batch[3]
    sl: Tensor = batch[4]
    if (tp <= 0).any() or (sl <= 0).any():
        prices: Tensor = batch[0]
        close: Tensor = prices[:, :, 3]  # close at index 3
        new_tp: Tensor = close + torch.abs(torch.randn_like(close)) * 2 + 1
        new_sl: Tensor = torch.maximum(
            close - torch.abs(torch.randn_like(close)) * 2 - 1,
            torch.ones_like(close),
        )
        batch[3] = new_tp.unsqueeze(-1)
        batch[4] = new_sl.unsqueeze(-1)
    # Convert action targets to float32 (validator expects float)
    batch[6] = batch[6].float()
    sample_batch = tuple(batch)
    validate_batch(
        sample_batch,
        expected_batch_size=sample_batch[0].shape[0],
        seq_len=sample_batch[0].shape[1],
        n_price_feats=sample_batch[0].shape[2],
        n_ind_feats=sample_batch[1].shape[2],
        n_sig_feats=sample_batch[2].shape[2],
        outcome_mode="binary",
    )
