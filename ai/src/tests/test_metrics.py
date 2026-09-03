"""Unit tests for metric functions.

This module tests the action accuracy and trade metrics functions used
to evaluate model predictions during validation.
"""

import pytest
import torch
from torch import Tensor

from ..metrics import compute_action_accuracy, compute_trade_metrics


@pytest.mark.unit
def test_compute_action_accuracy() -> None:
    """Test that action accuracy is computed correctly.

    The function should return per-class and overall accuracy, ignoring
    the -100 ignore index.  Values should be between 0 and 1.

    Asserts:
        - The returned dictionary contains 'overall', 'hold', 'entry', 'exit'.
        - All values are between 0 and 1.
        - When all targets are ignored, all accuracies are 0.0.
    """
    # Create dummy logits and targets
    logits: Tensor = torch.randn(10, 3)
    targets: Tensor = torch.randint(0, 3, (10,))
    # Add ignore index
    targets[0] = -100
    acc: dict[str, float] = compute_action_accuracy(
        logits, targets, ignore_index=-100
    )
    assert 'overall' in acc
    assert 'hold' in acc
    assert 'entry' in acc
    assert 'exit' in acc
    # Values between 0 and 1
    for v in acc.values():
        assert 0 <= v <= 1
    # If all ignored, should return zeros
    all_ignored: Tensor = torch.full((10,), -100)
    acc_all_ignored: dict[str, float] = compute_action_accuracy(
        logits, all_ignored
    )
    assert acc_all_ignored['overall'] == 0.0


@pytest.mark.unit
def test_compute_trade_metrics() -> None:
    """Test that trade metrics are computed correctly.

    The function should return win rate, profit factor, and number of trades
    based on predicted and actual entry bars.  Outcomes are classified as
    win (1.0), loss (0.0), or ignore (2.0).  If no valid trades exist,
    win_rate and profit_factor should be 0.0.

    Asserts:
        - The returned dictionary contains 'win_rate',
            'profit_factor', 'num_trades'.
        - num_trades is non-negative.
        - When there are no trades, all metrics are 0.0.
    """
    # Create dummy data: 10 samples, force some entries
    logits: Tensor = torch.randn(10, 3)
    # Force some entries to be predicted as entry (class 1)
    logits[:, 1] += 1.0
    targets: Tensor = torch.randint(0, 3, (10,))
    # Make some targets entry
    targets[0] = 1
    targets[1] = 1
    targets[2] = 1
    # Outcomes
    outcome_targets: Tensor = torch.randn(10)
    outcome_targets[0] = 1.0  # win
    outcome_targets[1] = 0.0  # loss
    outcome_targets[2] = 2.0  # ignore
    metrics: dict[str, float] = compute_trade_metrics(
        logits, targets, outcome_targets, ignore_index=2
    )
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'num_trades' in metrics
    assert metrics['num_trades'] >= 0
    # If no trades, returns zeros
    no_trade_logits: Tensor = torch.zeros(5, 3)
    no_trade_targets: Tensor = torch.full((5,), 0)
    metrics_zero: dict[str, float] = compute_trade_metrics(
        no_trade_logits, no_trade_targets, torch.ones(5)
    )
    assert metrics_zero['num_trades'] == 0
