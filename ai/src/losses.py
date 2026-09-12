"""Custom loss function for the EntryExitTransformer.

Combines cross-entropy for actions, an auxiliary outcome loss
(applied only on entry bars), and an optional multi-label pattern loss.
"""

import torch
import torch.nn.functional as functional


def dual_loss(
    action_logits: torch.Tensor,
    outcome_logits: torch.Tensor,
    action_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    outcome_mode: str = "binary",
    lambda_outcome: float = 0.3,
    ignore_index: int = 2,
    pattern_logits: torch.Tensor | None = None,
    pattern_targets: torch.Tensor | None = None,
    lambda_pattern: float = 0.1,
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss: action + lambda__outcome * outcome + lambda__pattern * pattern.

    The pattern loss is computed only when ``pattern_logits`` and
    ``pattern_targets`` are not None and have at least one feature.

    Args:
        action_logits: (B, T, 3) raw action logits.
        outcome_logits: Outcome logits, shape depends on mode.
        action_targets: (B, T) ground-truth action labels.
        outcome_targets: (B, T) ground-truth outcome labels.
        outcome_mode: 'binary', 'multiclass', or 'regression'.
        lambda_outcome: Weight for outcome loss (default 0.3).
        ignore_index: Value in outcome_targets to ignore (default 2).
        pattern_logits: (B, T, n_patterns) or None.
        pattern_targets: (B, T, n_patterns) or None.
        lambda_pattern: Weight for pattern loss (default 0.1).
        class_weight: Optional class weights for action cross-entropy.
            Tensor of shape (3,), passed to
            ``torch.nn.functional.cross_entropy``.

    Returns:
        tuple of four scalar tensors:
        - total_loss
        - action_loss
        - outcome_loss
        - pattern_loss

    """
    action_loss = functional.cross_entropy(
        action_logits.reshape(-1, action_logits.size(-1)),
        action_targets.reshape(-1),
        ignore_index=-100,
        weight=class_weight,
    )
    entry_mask = action_targets == 1
    if outcome_mode == "binary":
        logits = outcome_logits[entry_mask].squeeze(-1)
        targets = outcome_targets[entry_mask].float()
        valid = targets != ignore_index
        outcome_loss = (
            functional.binary_cross_entropy_with_logits(
                logits[valid], targets[valid]
            )
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    elif outcome_mode == "multiclass":
        logits = outcome_logits[entry_mask]
        targets = outcome_targets[entry_mask].long()
        valid = targets != ignore_index
        outcome_loss = (
            functional.cross_entropy(logits[valid], targets[valid])
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    elif outcome_mode == "regression":
        logits = outcome_logits[entry_mask].squeeze(-1)
        targets = outcome_targets[entry_mask].float()
        valid = ~torch.isnan(targets)
        outcome_loss = (
            functional.mse_loss(logits[valid], targets[valid])
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    else:
        outcome_loss = torch.tensor(0.0, device=action_logits.device)
    # Pattern loss
    if (
        pattern_logits is not None
        and pattern_targets is not None
        and pattern_targets.shape[-1] > 0
    ):
        pattern_loss = functional.binary_cross_entropy_with_logits(
            pattern_logits, pattern_targets
        )
    else:
        pattern_loss = torch.tensor(0.0, device=action_logits.device)
    total_loss = (
        action_loss
        + lambda_outcome * outcome_loss
        + lambda_pattern * pattern_loss
    )
    return total_loss, action_loss, outcome_loss, pattern_loss
