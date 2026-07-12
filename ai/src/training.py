"""Training and self-training pipelines for the EntryExitTransformer.

Provides:
- DataLoader builders for labeled and unlabeled Parquet data.
- A single-round supervised training function.
- Pseudo-label generation and self-training loop.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .dataset import TradingDataset, collate_ob
from .datatypes import OrderBlock
from .io import (
    load_features_parquet,
    load_labels_parquet,
    merge_features_labels,
    save_labels_parquet,
)
from .losses import dual_loss


def build_loader_from_parquet(
    features_path: str,
    labels_path: str,
    order_blocks: list[OrderBlock],
    seq_len: int,
    price_cols: list[str],
    ind_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    batch_size: int,
    shuffle: bool,
    pattern_cols: list[str] | None = None,
) -> tuple[DataLoader, pl.DataFrame]:
    """Create a DataLoader from feature and label Parquet files.

    Args:
        features_path: Path to features.parquet.
        labels_path: Path to labels.parquet.
        order_blocks: List of all order blocks.
        seq_len: Sequence length for windows.
        price_cols: Names of price columns.
        ind_cols: Names of indicator columns.
        sig_cols: Names of signal columns.
        tp_sl_cols: Names of TP/SL columns.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        pattern_cols: Optional list of pattern label column names.
            If provided, they are extracted as multi-label targets.

    Returns:
        A tuple (loader, merged_df) where merged_df is the joined
        features+labels DataFrame.

    """
    df_feat = load_features_parquet(features_path)
    df_lbl = load_labels_parquet(labels_path)
    df = merge_features_labels(df_feat, df_lbl)

    data = np.column_stack([
        df[price_cols].to_numpy(),
        df[ind_cols].to_numpy() if ind_cols else np.zeros((df.height, 0)),
        df[sig_cols].to_numpy() if sig_cols else np.zeros((df.height, 0)),
        df[tp_sl_cols].to_numpy(),
    ])

    action = df['action'].to_numpy()
    outcome = df['outcome'].to_numpy()

    if pattern_cols:
        pattern_targets = df[pattern_cols].to_numpy().astype(np.float32)
    else:
        pattern_targets = None

    dataset = TradingDataset(
        data=data,
        order_blocks=order_blocks,
        action_targets=action,
        outcome_targets=outcome,
        seq_len=seq_len,
        price_feats=len(price_cols),
        ind_feats=len(ind_cols),
        sig_feats=len(sig_cols),
        tp_sl_feats=len(tp_sl_cols),
        pattern_targets=pattern_targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ob,
    )
    return loader, df


def build_unlabeled_loader_from_parquet(
    features_path: str,
    order_blocks: list[OrderBlock],
    seq_len: int,
    price_cols: list[str],
    ind_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    batch_size: int,
    outcome_mode: str = 'binary',
) -> DataLoader:
    """Create a DataLoader for unlabeled data (all targets set to ignore).

    Args:
        features_path: Path to features.parquet (no labels required).
        order_blocks: List of all order blocks.
        seq_len: Sequence length.
        price_cols: Names of price columns.
        ind_cols: Names of indicator columns.
        sig_cols: Names of signal columns.
        tp_sl_cols: Names of TP/SL columns.
        batch_size: Batch size.
        outcome_mode: Used to decide the ignore value for outcomes.

    Returns:
        A DataLoader yielding batches with -100 actions and appropriate
        ignore values for outcomes. Pattern targets are empty.

    """
    df = load_features_parquet(features_path)
    n = df.height

    action = np.full(n, -100, dtype=int)
    if outcome_mode == 'regression':
        outcome = np.full(n, np.nan, dtype=float)
    else:
        outcome = np.full(n, 2, dtype=float)

    data = np.column_stack([
        df[price_cols].to_numpy(),
        df[ind_cols].to_numpy() if ind_cols else np.zeros((n, 0)),
        df[sig_cols].to_numpy() if sig_cols else np.zeros((n, 0)),
        df[tp_sl_cols].to_numpy(),
    ])

    dataset = TradingDataset(
        data=data,
        order_blocks=order_blocks,
        action_targets=action,
        outcome_targets=outcome,
        seq_len=seq_len,
        price_feats=len(price_cols),
        ind_feats=len(ind_cols),
        sig_feats=len(sig_cols),
        tp_sl_feats=len(tp_sl_cols),
        pattern_targets=None,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ob,
    )


def train_one_round(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
    lambda_pattern: float = 0.1,
):
    """Run a standard supervised training round.

    Args:
        model: The EntryExitTransformer model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of epochs.
        device: Torch device.
        outcome_mode: Passed to the loss function.
        lambda_outcome: Weight of the outcome loss.
        lr: Learning rate.
        lambda_pattern: Weight of the pattern loss (default 0.1).
            Ignored if pattern targets are empty.

    Returns:
        The trained model.

    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            (
                prices,
                indicators,
                signals,
                tp,
                sl,
                order_blocks,
                action_tgt,
                outcome_tgt,
                pattern_tgt,
            ) = batch
            prices = prices.to(device)
            indicators = indicators.to(device)
            signals = signals.to(device)
            tp = tp.to(device)
            sl = sl.to(device)
            action_tgt = action_tgt.to(device)
            outcome_tgt = outcome_tgt.to(device)
            pattern_tgt = pattern_tgt.to(device)

            optimizer.zero_grad()
            action_logits, outcome_logits, pattern_logits = model(
                prices, indicators, signals, tp, sl, order_blocks
            )
            loss, _, _, _ = dual_loss(
                action_logits,
                outcome_logits,
                action_tgt,
                outcome_tgt,
                outcome_mode,
                lambda_outcome,
                pattern_logits=pattern_logits,
                pattern_targets=pattern_tgt,
                lambda_pattern=lambda_pattern,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                (
                    prices,
                    indicators,
                    signals,
                    tp,
                    sl,
                    order_blocks,
                    action_tgt,
                    outcome_tgt,
                    pattern_tgt,
                ) = batch
                prices = prices.to(device)
                indicators = indicators.to(device)
                signals = signals.to(device)
                tp = tp.to(device)
                sl = sl.to(device)
                action_tgt = action_tgt.to(device)
                outcome_tgt = outcome_tgt.to(device)
                pattern_tgt = pattern_tgt.to(device)

                action_logits, outcome_logits, pattern_logits = model(
                    prices, indicators, signals, tp, sl, order_blocks
                )
                loss, _, _, _ = dual_loss(
                    action_logits,
                    outcome_logits,
                    action_tgt,
                    outcome_tgt,
                    outcome_mode,
                    lambda_outcome,
                    pattern_logits=pattern_logits,
                    pattern_targets=pattern_tgt,
                    lambda_pattern=lambda_pattern,
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(
            f'Epoch {epoch + 1}/{epochs} | '
            f'Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
        )
        scheduler.step(avg_val_loss)

    return model


def _check_rr_valid(
    entry_price: float,
    tp_price: float,
    sl_price: float,
    min_rr: float,
) -> bool:
    """Check if a potential entry meets the minimum risk-reward ratio.

    The direction (long/short) is inferred from relative TP/SL positions.

    Args:
        entry_price: Close price used as entry.
        tp_price: Take-profit level (absolute).
        sl_price: Stop-loss level (absolute).
        min_rr: Minimum required reward-to-risk ratio.

    Returns:
        True if the trade setup is valid and RR >= min_rr.

    """
    if entry_price > sl_price:  # long
        if not (sl_price < entry_price < tp_price):
            return False
        rr = (tp_price - entry_price) / (entry_price - sl_price)
    elif tp_price < entry_price < sl_price:
        rr = (entry_price - tp_price) / (sl_price - entry_price)
    else:
        return False
    return rr >= min_rr


def _determine_pseudo_outcome(
    outcome_logit: torch.Tensor,
    outcome_mode: str,
    outcome_threshold: float,
) -> float | None:
    """Convert raw outcome logit to a pseudo-outcome value if confident.

    Args:
        outcome_logit: The outcome logits tensor for a single bar
            (shape depends on mode).
        outcome_mode: 'binary', 'multiclass', or 'regression'.
        outcome_threshold: Confidence threshold.

    Returns:
        A float outcome value if the model is confident enough,
        otherwise None.

    """
    if outcome_mode == 'binary':
        prob = torch.sigmoid(outcome_logit)
        if prob > outcome_threshold:
            return 1.0
        return 0.0 if prob < (1 - outcome_threshold) else None
    elif outcome_mode == 'multiclass':
        probs = functional.softmax(outcome_logit, dim=-1)
        max_prob, cls = probs.max(dim=-1)
        return float(cls.item()) if max_prob >= outcome_threshold else None
    elif outcome_mode == 'regression':
        pred = outcome_logit.item()
        return float(pred) if abs(pred) >= outcome_threshold else None
    return None


def _generate_pseudo_labels_batch(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    outcome_mode: str,
    action_threshold: float,
    outcome_threshold: float,
    min_rr: float,
    close_idx: int,
    batch_idx: int,
    batch_size: int,
    seq_len: int,
) -> list[tuple[int, int, float]]:
    """Generate pseudo-labels for one batch of unlabeled data.

    For each bar where the current action label is ignore (-100),
    the model's prediction is examined.  If the action probability
    exceeds ``action_threshold`` and the outcome confidence meets
    ``outcome_threshold``, a pseudo-label is created.

    For entry actions (class 1), an additional risk-reward check
    is performed using the bar's TP/SL levels.

    Args:
        model: The model in eval mode.
        batch: A tuple as returned by the unlabeled DataLoader
            (9 elements, last is pattern_targets ignored here).
        device: Torch device.
        outcome_mode: See ``_determine_pseudo_outcome``.
        action_threshold: Minimum probability for a predicted action.
        outcome_threshold: Minimum confidence for outcome.
        min_rr: Minimum RR for entry pseudo-labels.
        close_idx: Index of the close price within the price tensor
            (usually 3 for OHLCV).
        batch_idx: Zero-based index of this batch.
        batch_size: Actual batch size (B).
        seq_len: Sequence length (T).

    Returns:
        A list of (global_index, pseudo_action, pseudo_outcome) tuples.

    """
    (
        prices,
        indicators,
        signals,
        tp,
        sl,
        order_blocks_batch,
        action_tgt,
        outcome_tgt,
        _pattern_tgt,
    ) = batch
    prices = prices.to(device)
    indicators = indicators.to(device)
    signals = signals.to(device)
    tp = tp.to(device)
    sl = sl.to(device)

    action_logits, outcome_logits, _pattern_logits = model(
        prices, indicators, signals, tp, sl, order_blocks_batch
    )
    action_probs = functional.softmax(action_logits, dim=-1)
    max_action_probs, pred_action = action_probs.max(dim=-1)

    b, t = prices.shape[0], prices.shape[1]
    new_pseudo = []

    for bi, ti in itertools.product(range(b), range(t)):
        if action_tgt[bi, ti] != -100:
            continue

        if (
            max_action_probs[bi, ti] <= action_threshold
            or pred_action[bi, ti] not in (1, 2)
        ):
            continue

        if pred_action[bi, ti] == 1:
            entry_price = prices[bi, ti, close_idx].item()
            tp_price = tp[bi, ti, 0].item()
            sl_price = sl[bi, ti, 0].item()
            if not _check_rr_valid(
                entry_price, tp_price, sl_price, min_rr
            ):
                continue

        pseudo_outcome = _determine_pseudo_outcome(
            outcome_logits[bi, ti], outcome_mode, outcome_threshold
        )
        if pseudo_outcome is None:
            continue

        pseudo_action = int(pred_action[bi, ti].item())
        global_idx = batch_idx * batch_size * t + bi * t + ti
        new_pseudo.append((global_idx, pseudo_action, pseudo_outcome))

    return new_pseudo


def _update_labels_parquet(
    labels_path: str,
    new_pseudo: list[tuple[int, int, float]],
    outcome_mode: str = 'binary',
) -> None:
    """Update the labels Parquet file with new pseudo-labels.

    Loads the existing labels file, modifies the action and outcome
    arrays at the specified global indices, and saves the result.

    Args:
        labels_path: Path to labels.parquet.
        new_pseudo: List of (global_idx, action, outcome) pseudo-labels.
        outcome_mode: Used to determine default outcome fill value.

    """
    df_lbl = load_labels_parquet(labels_path)

    if 'action' not in df_lbl.columns:
        df_lbl = df_lbl.with_columns([
            pl.lit(-100).alias('action'),
            pl.lit(
                float('nan') if outcome_mode == 'regression' else 2.0
            ).alias('outcome'),
        ])

    action_arr = df_lbl['action'].to_numpy()
    outcome_arr = df_lbl['outcome'].to_numpy()

    for global_idx, pseudo_action, pseudo_outcome in new_pseudo:
        if 0 <= global_idx < len(action_arr):
            action_arr[global_idx] = pseudo_action
            outcome_arr[global_idx] = pseudo_outcome

    df_lbl = df_lbl.with_columns([
        pl.Series('action', action_arr),
        pl.Series('outcome', outcome_arr),
    ])
    save_labels_parquet(df_lbl, labels_path)


def self_training_loop(
    model: torch.nn.Module,
    features_path: str,
    labels_path: str,
    features_path_unlabeled: str,
    order_blocks: list[OrderBlock],
    price_cols: list[str],
    ind_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
    epochs_per_round: int = 3,
    num_rounds: int = 3,
    action_threshold: float = 0.9,
    outcome_threshold: float = 0.8,
    min_rr: float = 1 / 3,
    close_idx: int = 3,
    save_model_path: str | None = None,
):
    """Run iterative self-training using a separate pool of unlabeled data.

    Each round:
    1. Train on the current labeled set.
    2. Generate pseudo-labels on the unlabeled set.
    3. Add confident pseudo-labels to the labeled set.
    4. (Optionally) save a model checkpoint.

    Args:
        model: The EntryExitTransformer to train.
        features_path: Labeled features.parquet.
        labels_path: Labels.parquet (will be updated in-place).
        features_path_unlabeled: Unlabeled features.parquet.
        order_blocks: List of all order blocks.
        price_cols: Names of price columns.
        ind_cols: Names of indicator columns.
        sig_cols: Names of signal columns.
        tp_sl_cols: Names of TP/SL columns.
        seq_len: Sequence length.
        batch_size: Batch size for both loaders.
        device: Torch device.
        outcome_mode: Outcome prediction mode.
        lambda_outcome: Outcome loss weight.
        lr: Learning rate.
        epochs_per_round: Training epochs per round.
        num_rounds: Maximum number of self-training rounds.
        action_threshold: Confidence threshold for action.
        outcome_threshold: Confidence threshold for outcome.
        min_rr: Minimum RR for entry pseudo-labels.
        close_idx: Index of close price in price features.
        save_model_path: If not None, saves model after each round with
            suffix '_roundN.pt'.

    Returns:
        The trained model.

    """
    labeled_loader, _ = build_loader_from_parquet(
        features_path,
        labels_path,
        order_blocks,
        seq_len,
        price_cols,
        ind_cols,
        sig_cols,
        tp_sl_cols,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = labeled_loader

    unlabeled_loader = build_unlabeled_loader_from_parquet(
        features_path_unlabeled,
        order_blocks,
        seq_len,
        price_cols,
        ind_cols,
        sig_cols,
        tp_sl_cols,
        batch_size=batch_size,
        outcome_mode=outcome_mode,
    )

    for round_idx in range(num_rounds):
        print(f'Self-training round {round_idx + 1}/{num_rounds}')

        model = train_one_round(
            model,
            labeled_loader,
            val_loader,
            epochs_per_round,
            device,
            outcome_mode,
            lambda_outcome,
            lr,
        )

        model.eval()
        all_new_pseudo = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                batch_pseudo = _generate_pseudo_labels_batch(
                    model,
                    batch,
                    device,
                    outcome_mode,
                    action_threshold,
                    outcome_threshold,
                    min_rr,
                    close_idx,
                    batch_idx,
                    batch_size,
                    seq_len,
                )
                all_new_pseudo.extend(batch_pseudo)

        if not all_new_pseudo:
            print('No new pseudo-labels, stopping self-training.')
            break
        print(
            f'Generated {len(all_new_pseudo)} pseudo-labels. '
            'Updating labels.parquet...'
        )

        _update_labels_parquet(labels_path, all_new_pseudo, outcome_mode)

        labeled_loader, _ = build_loader_from_parquet(
            features_path,
            labels_path,
            order_blocks,
            seq_len,
            price_cols,
            ind_cols,
            sig_cols,
            tp_sl_cols,
            batch_size=batch_size,
            shuffle=True,
        )

        if save_model_path:
            torch.save(
                model.state_dict(),
                f'{save_model_path}_round{round_idx + 1}.pt',
            )

    return model
