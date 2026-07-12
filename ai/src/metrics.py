"""Validation metrics for the EntryExitTransformer."""

from __future__ import annotations

import torch


def compute_action_accuracy(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Calculate per‑class accuracy and overall accuracy for actions.

    Args:
        action_logits: (N, 3) raw logits (flattened).
        action_targets: (N,) long targets with ignore_index.
        ignore_index: Value to ignore (default -100).

    Returns:
        Dict with keys 'overall', 'hold', 'entry', 'exit'.

    """
    mask = action_targets != ignore_index
    if not mask.any():
        return {'overall': 0.0, 'hold': 0.0, 'entry': 0.0, 'exit': 0.0}

    preds = action_logits.argmax(dim=-1)
    targets = action_targets[mask]
    correct = (preds[mask] == targets).float()

    overall = correct.mean().item()
    results = {'overall': overall}
    for cls, name in enumerate(['hold', 'entry', 'exit']):
        cls_mask = targets == cls
        results[name] = (
            correct[cls_mask].mean().item()
            if cls_mask.any() else 0.0
        )
    return results


def compute_trade_metrics(
    action_logits: torch.Tensor,
    outcome_logits: torch.Tensor,
    action_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    tp_levels: torch.Tensor,
    sl_levels: torch.Tensor,
    prices: torch.Tensor,
    close_idx: int = 3,
    ignore_index: int = 2,
) -> dict[str, float]:
    """Simulate trades on validation set and compute win rate & profit factor.

    Uses a simple rule: if predicted action is entry (1) and ground‑truth
    action is also entry, assume we enter at the bar's close.  Exit is
    determined by TP/SL, and outcome by ground‑truth (binary) or predicted
    (if regression, we skip).

    This is a *rough* estimate intended for monitoring, not backtesting.

    Args:
        action_logits: (B, T, 3)
        outcome_logits: (B, T, …)
        action_targets: (B, T)
        outcome_targets: (B, T)
        tp_levels: (B, T, 1)
        sl_levels: (B, T, 1)
        prices: (B, T, price_feats) - close is at close_idx.
        close_idx: Index of close price within price features.
        ignore_index: Ignore value for outcome (default 2 for binary).

    Returns:
        Dict with 'win_rate', 'profit_factor', 'num_trades'.

    """
    action_pred = action_logits.argmax(dim=-1)          # (B,T)
    action_true = action_targets
    # Берем только бары, где модель предсказала entry и истина = entry
    entry_mask = (action_pred == 1) & (action_true == 1)
    if not entry_mask.any():
        return {'win_rate': 0.0, 'profit_factor': 0.0, 'num_trades': 0}
    # Собираем предсказанные исходы (для простоты используем ground truth,
    # т.к. у нас есть outcome_targets)
    # В бинарном режиме 1 = win, 0 = loss.
    outcome_true = outcome_targets[entry_mask].float()
    # Игнорируем ignore_index (2)
    valid = outcome_true != ignore_index
    if not valid.any():
        return {'win_rate': 0.0, 'profit_factor': 0.0, 'num_trades': 0}
    wins = (outcome_true[valid] == 1.0).sum().item()
    losses = (outcome_true[valid] == 0.0).sum().item()
    total = wins + losses
    win_rate = wins / total if total else 0.0
    # Profit factor: сумма выигрышей / сумма проигрышей (в R)
    # Здесь у нас только бинарный outcome, поэтому profit factor = wins/losses
    profit_factor = wins / losses if losses > 0 else float('inf')
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': total,
    }
