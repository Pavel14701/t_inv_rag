"""Unit tests for the dual_loss function.

The dual_loss combines action cross-entropy, outcome loss (binary, multiclass,
or regression), and an optional multi-label pattern loss.  This module tests
all three outcome modes and pattern loss integration.
"""

import pytest
import torch

from torch import Tensor

from ai.src.losses import dual_loss


@pytest.mark.unit
def test_dual_loss_binary() -> None:
    """Test dual_loss in binary outcome mode with ignore indices.

    The loss should be a scalar >= 0.  Action targets include an ignore
    value (-100) which should be excluded from the action cross-entropy.
    Outcome targets include an ignore value (2) which should be excluded
    from the binary cross-entropy.

    Asserts:
        - total_loss, action_loss, outcome_loss, pattern_loss are not None.
        - total_loss is a scalar (dimension 0) and >= 0.
    """
    batch_size, seq_len = 2, 10
    action_logits: Tensor = torch.randn(batch_size, seq_len, 3)
    outcome_logits: Tensor = torch.randn(batch_size, seq_len, 1)
    action_targets: Tensor = torch.randint(0, 3, (batch_size, seq_len))
    action_targets[0, 0] = -100  # ignore
    outcome_targets: Tensor = torch.randint(
        0, 3, (batch_size, seq_len)
    ).float()
    outcome_targets[0, 1] = 2  # ignore for outcome
    total_loss, action_loss, outcome_loss, pattern_loss = dual_loss(
        action_logits,
        outcome_logits,
        action_targets,
        outcome_targets,
        outcome_mode="binary",
        lambda_outcome=0.3,
    )
    assert total_loss is not None
    assert action_loss is not None
    assert outcome_loss is not None
    assert pattern_loss is not None
    assert total_loss >= 0
    assert total_loss.dim() == 0  # scalar


@pytest.mark.unit
def test_dual_loss_multiclass() -> None:
    """Test dual_loss in multiclass outcome mode.

    Outcome logits have 3 classes, and targets are integer class labels.
    The ignore value (2) should be excluded from the multiclass cross-entropy.

    Asserts:
        - total_loss is a scalar >= 0.
    """
    batch_size, seq_len = 2, 10
    action_logits: Tensor = torch.randn(batch_size, seq_len, 3)
    outcome_logits: Tensor = torch.randn(batch_size, seq_len, 3)  # 3 classes
    action_targets: Tensor = torch.randint(0, 3, (batch_size, seq_len))
    outcome_targets: Tensor = torch.randint(0, 3, (batch_size, seq_len))
    outcome_targets[0, 2] = 2  # ignore
    total_loss, _action_loss, _outcome_loss, _pattern_loss = dual_loss(
        action_logits,
        outcome_logits,
        action_targets,
        outcome_targets,
        outcome_mode="multiclass",
        lambda_outcome=0.3,
    )
    assert total_loss >= 0


@pytest.mark.unit
def test_dual_loss_regression() -> None:
    """Test dual_loss in regression outcome mode.

    Outcome targets are continuous values, and the loss is Mean Squared Error
    between the logits and targets (ignoring NaN values if present).

    Asserts:
        - total_loss is a scalar >= 0.
    """
    batch_size, seq_len = 2, 10
    action_logits: Tensor = torch.randn(batch_size, seq_len, 3)
    outcome_logits: Tensor = torch.randn(batch_size, seq_len, 1)
    action_targets: Tensor = torch.randint(0, 3, (batch_size, seq_len))
    # regression targets
    outcome_targets: Tensor = torch.randn(batch_size, seq_len)
    total_loss, _action_loss, _outcome_loss, _pattern_loss = dual_loss(
        action_logits,
        outcome_logits,
        action_targets,
        outcome_targets,
        outcome_mode="regression",
        lambda_outcome=0.3,
    )
    assert total_loss >= 0


@pytest.mark.unit
def test_dual_loss_with_pattern() -> None:
    """Test dual_loss with pattern loss enabled.

    Pattern logits and targets are provided, and lambda_pattern > 0.
    The pattern loss should be >= 0 and contribute to the total loss.

    Asserts:
        - pattern_loss is >= 0.
        - total_loss is greater than 0 (since at least action_loss > 0).
    """
    batch_size, seq_len = 2, 10
    n_patterns = 5
    action_logits: Tensor = torch.randn(batch_size, seq_len, 3)
    outcome_logits: Tensor = torch.randn(batch_size, seq_len, 1)
    pattern_logits: Tensor = torch.randn(batch_size, seq_len, n_patterns)
    pattern_targets: Tensor = torch.randint(
        0, 2, (batch_size, seq_len, n_patterns)
    ).float()
    action_targets: Tensor = torch.randint(0, 3, (batch_size, seq_len))
    outcome_targets: Tensor = torch.randint(
        0, 3, (batch_size, seq_len)
    ).float()
    total_loss, _action_loss, _outcome_loss, pattern_loss = dual_loss(
        action_logits,
        outcome_logits,
        action_targets,
        outcome_targets,
        outcome_mode="binary",
        lambda_outcome=0.3,
        pattern_logits=pattern_logits,
        pattern_targets=pattern_targets,
        lambda_pattern=0.1,
    )
    assert pattern_loss >= 0
    assert total_loss > 0
