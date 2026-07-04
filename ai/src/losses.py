import torch
import torch.nn.functional as F


def dual_loss(
    action_logits: torch.Tensor,
    outcome_logits: torch.Tensor,
    action_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    ignore_index: int = 2,
):
    action_loss = F.cross_entropy(
        action_logits.reshape(-1, action_logits.size(-1)),
        action_targets.reshape(-1),
        ignore_index=-100,
    )

    entry_mask = action_targets == 1
    if outcome_mode == 'binary':
        logits = outcome_logits[entry_mask].squeeze(-1)
        targets = outcome_targets[entry_mask].float()
        valid = targets != ignore_index
        outcome_loss = (
            F.binary_cross_entropy_with_logits(logits[valid], targets[valid])
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    elif outcome_mode == 'multiclass':
        logits = outcome_logits[entry_mask]
        targets = outcome_targets[entry_mask].long()
        valid = targets != ignore_index
        outcome_loss = (
            F.cross_entropy(logits[valid], targets[valid])
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    elif outcome_mode == 'regression':
        logits = outcome_logits[entry_mask].squeeze(-1)
        targets = outcome_targets[entry_mask].float()
        valid = ~torch.isnan(targets)
        outcome_loss = (
            F.mse_loss(logits[valid], targets[valid])
            if valid.sum() > 0
            else torch.tensor(0.0, device=action_logits.device)
        )
    else:
        outcome_loss = torch.tensor(0.0, device=action_logits.device)

    total_loss = action_loss + lambda_outcome * outcome_loss
    return total_loss, action_loss, outcome_loss
