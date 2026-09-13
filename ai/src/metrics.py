"""Validation metrics for the EntryExitTransformer."""

from __future__ import annotations

import torch


def compute_action_accuracy(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Calculate per-class accuracy and overall accuracy for actions.

    Args:
        action_logits: (N, 3) raw logits (flattened).
        action_targets: (N,) long targets with ignore_index.
        ignore_index: Value to ignore (default -100).

    Returns:
        Dict with keys 'overall', 'hold', 'entry', 'exit'.

    """
    mask = action_targets != ignore_index
    if not mask.any():
        return {"overall": 0.0, "hold": 0.0, "entry": 0.0, "exit": 0.0}

    preds = action_logits.argmax(dim=-1)
    targets = action_targets[mask]
    correct = (preds[mask] == targets).float()

    overall = correct.mean().item()
    results = {"overall": overall}
    for cls, name in enumerate(["hold", "entry", "exit"]):
        cls_mask = targets == cls
        results[name] = (
            correct[cls_mask].mean().item() if cls_mask.any() else 0.0
        )
    return results


def compute_trade_metrics(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    ignore_index: int = 2,
) -> dict[str, float]:
    """Simple trade-like metrics based on the model's predicted entries.

    Only bars where the model **predicts** an entry (``pred == 1``) and
    the ground-truth outcome is known are considered.  Restricting the
    metric to bars where the ground truth also says ``entry`` would
    bias the Win Rate towards the label generator's choices.

    Args:
        action_logits: (N, 3) flattened action logits.
        action_targets: (N,) long action labels (unused for masking,
            kept for API compatibility).
        outcome_targets: (N,) float outcome labels.
        ignore_index: Value in outcome_targets to ignore (default 2).

    Returns:
        Dict with 'win_rate', 'profit_factor', 'num_trades'.

    """
    action_pred = action_logits.argmax(dim=-1)
    entry_mask = action_pred == 1
    if not entry_mask.any():
        return {"win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0}

    outcome_true = outcome_targets[entry_mask].float()
    valid = outcome_true != ignore_index
    if not valid.any():
        return {"win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0}

    wins = (outcome_true[valid] == 1.0).sum().item()  # noqa: RUF069 - exact IEEE zero/sign check
    losses = (outcome_true[valid] == 0.0).sum().item()  # noqa: RUF069 - exact IEEE zero/sign check
    total = wins + losses
    win_rate = wins / total if total else 0.0
    profit_factor = wins / losses if losses > 0 else float("inf")

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": total,
    }
