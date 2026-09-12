"""Unit tests for the EntryExitTransformer model.

This module tests the transformer model's forward pass in different modes
(binary, multiclass, regression) and verifies shape correctness and
device movement.
"""

import pytest
import torch
from typing import Any

from ..transformer import EntryExitTransformer


@pytest.mark.unit
def test_transformer_forward(
    model_params: dict[str, Any],
    sample_batch_tensors: dict[str, Any],
) -> None:
    """Test the forward pass of EntryExitTransformer in binary outcome mode.

    The model should produce action, outcome, and pattern logits with
    correct shapes.  Outcome logits shape is (B, T, 1) for binary mode.

    Args:
        model_params: Fixture with model hyperparameters.
        sample_batch_tensors: Fixture with input tensors.

    Asserts:
        - Action logits shape: (B, T, 3)
        - Outcome logits shape: (B, T, 1)
        - Pattern logits shape: (B, T, n_patterns)

    """
    model = EntryExitTransformer(**model_params)
    batch = sample_batch_tensors
    action_logits, outcome_logits, pattern_logits = model(
        batch['prices'],
        batch['indicators'],
        batch['signals'],
        batch['tp'],
        batch['sl'],
        batch['order_blocks'],
    )
    B, T = batch['prices'].shape[0], batch['prices'].shape[1]  # noqa: N806
    assert action_logits.shape == (B, T, 3)
    assert outcome_logits.shape == (B, T, 1)  # binary mode
    assert pattern_logits.shape == (B, T, model_params['n_patterns'])


@pytest.mark.unit
def test_transformer_outcome_multiclass(
    model_params: dict[str, Any],
    sample_batch_tensors: dict[str, Any],
) -> None:
    """Test the forward pass with multiclass outcome mode.

    The outcome head should produce logits of shape (B, T, n_outcome_classes).

    Args:
        model_params: Fixture with model hyperparameters.
        sample_batch_tensors: Fixture with input tensors.

    Asserts:
        - Outcome logits shape: (B, T, 3) when n_outcome_classes=3.

    """
    params = model_params.copy()
    params['outcome_mode'] = 'multiclass'
    params['n_outcome_classes'] = 3
    model = EntryExitTransformer(**params)
    batch = sample_batch_tensors
    _, outcome_logits, _ = model(
        batch['prices'],
        batch['indicators'],
        batch['signals'],
        batch['tp'],
        batch['sl'],
        batch['order_blocks'],
    )
    B, T = batch['prices'].shape[0], batch['prices'].shape[1]  # noqa: N806
    assert outcome_logits.shape == (B, T, 3)


@pytest.mark.unit
def test_transformer_outcome_regression(
    model_params: dict[str, Any],
    sample_batch_tensors: dict[str, Any],
) -> None:
    """Test the forward pass with regression outcome mode.

    The outcome head should produce logits of shape (B, T, 1).

    Args:
        model_params: Fixture with model hyperparameters.
        sample_batch_tensors: Fixture with input tensors.

    Asserts:
        - Outcome logits shape: (B, T, 1).

    """
    params = model_params.copy()
    params['outcome_mode'] = 'regression'
    model = EntryExitTransformer(**params)
    batch = sample_batch_tensors
    _, outcome_logits, _ = model(
        batch['prices'],
        batch['indicators'],
        batch['signals'],
        batch['tp'],
        batch['sl'],
        batch['order_blocks'],
    )
    B, T = batch['prices'].shape[0], batch['prices'].shape[1]  # noqa: N806
    assert outcome_logits.shape == (B, T, 1)


@pytest.mark.unit
def test_transformer_device_movement(
    model_params: dict[str, Any],
    sample_batch_tensors: dict[str, Any],
) -> None:
    """Test that the model correctly moves tensors to the GPU if available.

    The test checks that after moving the model and inputs to CUDA,
    all output tensors reside on the CUDA device.

    Args:
        model_params: Fixture with model hyperparameters.
        sample_batch_tensors: Fixture with input tensors.

    Skips:
        If CUDA is not available, the test is skipped.

    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    model = EntryExitTransformer(**model_params).to('cuda')
    batch = sample_batch_tensors
    # Move inputs to CUDA
    prices = batch['prices'].to('cuda')
    indicators = batch['indicators'].to('cuda')
    signals = batch['signals'].to('cuda')
    tp = batch['tp'].to('cuda')
    sl = batch['sl'].to('cuda')
    action_logits, outcome_logits, pattern_logits = model(
        prices, indicators, signals, tp, sl, batch['order_blocks']
    )
    assert action_logits.device.type == 'cuda'
    assert outcome_logits.device.type == 'cuda'
    assert pattern_logits.device.type == 'cuda'
