"""Training and self-training pipelines for the EntryExitTransformer.

Provides:
- DataLoader builders for labeled and unlabeled Parquet data.
- A single-round supervised training function with validation split,
    metrics, checkpointing, and early stopping.
- Pseudo-label generation and self-training loop with correct global
    bar index alignment.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import polars as pl
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .dataset import TradingDataset, collate_ob
from .datatypes import OrderBlock
from .io import (
    load_features_parquet,
    load_labels_parquet,
    merge_features_labels,
    save_labels_parquet,
)
from .losses import dual_loss
from .metrics import compute_action_accuracy, compute_trade_metrics


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
    bar_index_col: str | None = 'bar_index',
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
        bar_index_col: Column name containing a stable bar identifier.
            If present in the data, it is used for safe pseudo-label
            alignment in self-training. If ``None`` or missing,
            positional indices are used.

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

    if bar_index_col and bar_index_col in df.columns:
        bar_index = df[bar_index_col].to_numpy().astype(np.int64)
    else:
        bar_index = None

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
        bar_index=bar_index,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ob,
    )


def _compute_class_weights(action_targets: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency weights for action classes (ignore -100).

    Args:
        action_targets: 1D array of integer labels.

    Returns:
        Float tensor of shape (3,) with class weights normalised so
        that the mean weight is 1.

    """
    unique, counts = np.unique(
        action_targets[action_targets != -100], return_counts=True
    )
    weights = np.ones(3, dtype=np.float32)
    for cls, cnt in zip(unique, counts):
        if cls in (0, 1, 2):
            weights[int(cls)] = 1.0 / cnt
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _prepare_batch(batch, device: torch.device):
    """Unpack a 10-element batch and move tensors to device.

    Returns a dict with all tensor fields + order_blocks (list).
    """
    (
        prices, indicators, signals, tp, sl,
        order_blocks, action_tgt, outcome_tgt,
        pattern_tgt, start_indices,
    ) = batch
    return {
        'prices': prices.to(device),
        'indicators': indicators.to(device),
        'signals': signals.to(device),
        'tp': tp.to(device),
        'sl': sl.to(device),
        'order_blocks': order_blocks,          # list of lists, stays on CPU
        'action_tgt': action_tgt.to(device),
        'outcome_tgt': outcome_tgt.to(device),
        'pattern_tgt': pattern_tgt.to(device),
        'start_indices': start_indices,
    }


def _model_forward_loss(
    model: torch.nn.Module,
    batch_data: dict,
    outcome_mode: str,
    lambda_outcome: float,
    lambda_pattern: float,
    class_weight: torch.Tensor | None,
):
    """Run forward pass and compute loss.

    Returns (loss, action_logits, outcome_logits).
    """
    action_logits, outcome_logits, pattern_logits = model(
        batch_data['prices'],
        batch_data['indicators'],
        batch_data['signals'],
        batch_data['tp'],
        batch_data['sl'],
        batch_data['order_blocks'],
    )
    loss, _, _, _ = dual_loss(
        action_logits,
        outcome_logits,
        batch_data['action_tgt'],
        batch_data['outcome_tgt'],
        outcome_mode,
        lambda_outcome,
        pattern_logits=pattern_logits,
        pattern_targets=batch_data['pattern_tgt'],
        lambda_pattern=lambda_pattern,
        class_weight=class_weight,
    )
    return loss, action_logits, outcome_logits


def _compute_val_metrics(
    all_action_logits: list[torch.Tensor],
    all_action_targets: list[torch.Tensor],
    all_outcome_logits: list[torch.Tensor],
    all_outcome_targets: list[torch.Tensor],
    all_prices: list[torch.Tensor],
    all_tp: list[torch.Tensor],
    all_sl: list[torch.Tensor],
    outcome_mode: str,
    close_idx: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Concatenate accumulated tensors and compute
    action accuracy & trade metrics.

    Returns:
        (action_acc_dict, trade_metrics_dict)

    """
    cat_action_logits = torch.cat(all_action_logits)
    cat_action_targets = torch.cat(all_action_targets)
    action_acc = compute_action_accuracy(cat_action_logits, cat_action_targets)

    # Trade metrics – may fail if no entry bars, handle gracefully
    try:
        cat_prices = torch.cat(all_prices, dim=0)
        cat_tp = torch.cat(all_tp, dim=0)
        cat_sl = torch.cat(all_sl, dim=0)
        cat_outcome_logits = torch.cat(all_outcome_logits, dim=0)
        cat_outcome_targets = torch.cat(all_outcome_targets, dim=0)

        trade_metrics = compute_trade_metrics(
            cat_prices.unsqueeze(0),
            cat_outcome_logits.unsqueeze(0),
            cat_action_targets.reshape(1, -1),
            cat_outcome_targets.reshape(1, -1),
            cat_tp.unsqueeze(0),
            cat_sl.unsqueeze(0),
            cat_prices.unsqueeze(0),
            close_idx=close_idx,
        )
    except RuntimeError:
        # Typically happens when shapes cannot be concatenated
        trade_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0
        }
    return action_acc, trade_metrics


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    outcome_mode: str,
    lambda_outcome: float,
    lambda_pattern: float,
    class_weight: torch.Tensor | None,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        data = _prepare_batch(batch, device)
        loss, _, _ = _model_forward_loss(
            model, data, outcome_mode, lambda_outcome,
            lambda_pattern, class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def _run_val_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    outcome_mode: str,
    lambda_outcome: float,
    lambda_pattern: float,
    close_idx: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Validate for one epoch.

    Returns:
        avg_val_loss, action_acc, trade_metrics

    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    # Accumulators
    a_logits, a_targets = [], []
    o_logits, o_targets = [], []
    prices_list, tp_list, sl_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            data = _prepare_batch(batch, device)
            loss, action_logits, outcome_logits = _model_forward_loss(
                model, data, outcome_mode, lambda_outcome,
                lambda_pattern, None
            )
            total_loss += loss.item()
            num_batches += 1

            # Flatten and store
            a_logits.append(action_logits.reshape(-1, 3))
            a_targets.append(data['action_tgt'].reshape(-1))
            o_logits.append(
                outcome_logits.reshape(
                    -1,
                    (
                        outcome_logits.size(-1)
                        if outcome_mode == 'multiclass'
                        else 1
                    )
                )
            )
            o_targets.append(data['outcome_tgt'].reshape(-1))
            prices_list.append(data['prices'])
            tp_list.append(data['tp'])
            sl_list.append(data['sl'])

    avg_loss = total_loss / max(num_batches, 1)
    action_acc, trade_metrics = _compute_val_metrics(
        a_logits, a_targets,
        o_logits, o_targets,
        prices_list, tp_list, sl_list,
        outcome_mode, close_idx,
    )
    return avg_loss, action_acc, trade_metrics


def _log_epoch(
    writer: SummaryWriter | None,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    action_acc: dict[str, float],
    trade_metrics: dict[str, float],
):
    """Print console message and write TensorBoard scalars."""
    msg = (
        f'Epoch {epoch + 1}/{total_epochs} | '
        f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
        f'Acc: {action_acc["overall"]:.3f} (entry: {action_acc["entry"]:.3f}) | '  # noqa: E501
        f'WR: {trade_metrics["win_rate"]:.3f} PF: {trade_metrics["profit_factor"]:.2f}'  # noqa: E501
    )
    print(msg)

    if writer is not None:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/overall', action_acc['overall'], epoch)
        writer.add_scalar('Acc/entry', action_acc['entry'], epoch)
        writer.add_scalar('Trades/win_rate', trade_metrics['win_rate'], epoch)
        writer.add_scalar(
            'Trades/profit_factor',
            trade_metrics['profit_factor'],
            epoch
        )


def _checkpoint_and_stop(
    val_loss: float,
    best_val_loss: float,
    no_improve_count: int,
    model: torch.nn.Module,
    save_best: bool,
    best_model_path: str | None,
    early_stopping_patience: int,
) -> tuple[float, int, bool]:
    """Update best loss, save model, and decide whether to stop.

    Returns:
        (updated_best_val_loss, updated_no_improve_count, should_stop)

    """
    if save_best and val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        if best_model_path is not None:
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> Best model saved (val_loss={best_val_loss:.4f})')
    else:
        no_improve_count += 1

    should_stop = (
        early_stopping_patience > 0
        and no_improve_count >= early_stopping_patience
    )
    if should_stop:
        print('  Early stopping.')
    return best_val_loss, no_improve_count, should_stop


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
    class_weight: torch.Tensor | None = None,
    log_dir: str | None = None,
    save_best: bool = True,
    best_model_path: str | None = None,
    early_stopping_patience: int = 0,
    close_idx: int = 3,
):
    """Train the model, now with minimal code in the main loop."""
    writer = SummaryWriter(log_dir) if log_dir else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    best_val_loss = math.inf
    no_improve_count = 0

    for epoch in range(epochs):
        train_loss = _run_train_epoch(
            model, train_loader, device, optimizer,
            outcome_mode, lambda_outcome, lambda_pattern, class_weight
        )
        val_loss, action_acc, trade_metrics = _run_val_epoch(
            model, val_loader, device,
            outcome_mode, lambda_outcome, lambda_pattern, close_idx
        )
        _log_epoch(
            writer, epoch, epochs,
            train_loss, val_loss, action_acc,
            trade_metrics
        )
        scheduler.step(val_loss)

        best_val_loss, no_improve_count, stop = _checkpoint_and_stop(
            val_loss, best_val_loss, no_improve_count, model,
            save_best, best_model_path, early_stopping_patience
        )
        if stop:
            break

    if writer:
        writer.close()
    return model


def _check_rr_valid(
    entry_price: float,
    tp_price: float,
    sl_price: float,
    min_rr: float,
) -> bool:
    """Check if a potential entry meets the minimum risk-reward ratio."""
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
    """Convert raw outcome logit to pseudo-outcome if confident."""
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
    seq_len: int,
) -> list[tuple[int, int, float]]:
    (
        prices, indicators, signals, tp, sl,
        order_blocks_batch, action_tgt, outcome_tgt,
        _pattern_tgt, start_indices, bar_indices,
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
            if not _check_rr_valid(entry_price, tp_price, sl_price, min_rr):
                continue

        pseudo_outcome = _determine_pseudo_outcome(
            outcome_logits[bi, ti], outcome_mode, outcome_threshold
        )
        if pseudo_outcome is None:
            continue

        pseudo_action = int(pred_action[bi, ti].item())
        # Глобальный индекс берётся из стабильного bar_indices
        global_bar = int(bar_indices[bi].item()) + ti
        new_pseudo.append((global_bar, pseudo_action, pseudo_outcome))

    return new_pseudo


def _update_labels_parquet(
    labels_path: str,
    new_pseudo: list[tuple[int, int, float]],
    outcome_mode: str = 'binary',
) -> None:
    """Update the labels Parquet file with new pseudo-labels.

    Args:
        labels_path: Path to labels.parquet.
        new_pseudo: List of (global_bar_index, action, outcome) tuples.
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
    n_rows = len(action_arr)

    for global_bar, pseudo_action, pseudo_outcome in new_pseudo:
        if 0 <= global_bar < n_rows:
            action_arr[global_bar] = pseudo_action
            outcome_arr[global_bar] = pseudo_outcome

    df_lbl = df_lbl.with_columns([
        pl.Series('action', action_arr),
        pl.Series('outcome', outcome_arr),
    ])
    save_labels_parquet(df_lbl, labels_path)


def _split_train_val(
    loader: DataLoader,
    val_split: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Split the dataset of a loader into training and validation loaders.

    Args:
        loader: Loader whose dataset is a TradingDataset.
        val_split: Fraction of data to use for validation (0 < val_split < 1).
        batch_size: Batch size for both returned loaders.

    Returns:
        train_loader, val_loader

    """
    dataset = loader.dataset
    # Explicitly confirm it's a TradingDataset so that len() is safe.
    assert isinstance(dataset, TradingDataset), (
        'loader.dataset must be a TradingDataset'
    )
    if val_split <= 0:
        return loader, loader

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    if n_val == 0:
        raise ValueError('val_split too small, validation set is empty')
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ob,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ob,
    )
    return train_loader, val_loader


def _self_training_round(
    model: torch.nn.Module,
    round_idx: int,
    num_rounds: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    unlabeled_loader: DataLoader,
    device: torch.device,
    outcome_mode: str,
    lambda_outcome: float,
    lr: float,
    epochs_per_round: int,
    action_threshold: float,
    outcome_threshold: float,
    min_rr: float,
    close_idx: int,
    seq_len: int,
    class_weight: torch.Tensor | None,
    log_dir: str | None,
    save_best: bool,
    best_model_path: str | None,
    early_stopping_patience: int,
) -> list[tuple[int, int, float]] | None:
    """Execute one round of self-training.

    Returns:
        List of newly generated pseudo-labels, or None if no new labels.

    """
    print(f'\n=== Self-training round {round_idx + 1}/{num_rounds} ===')
    model = train_one_round(
        model,
        train_loader,
        val_loader,
        epochs_per_round,
        device,
        outcome_mode,
        lambda_outcome,
        lr,
        lambda_pattern=0.1,  # pattern loss unused in self-training
        class_weight=class_weight,
        log_dir=log_dir,
        save_best=save_best,
        best_model_path=best_model_path,
        early_stopping_patience=early_stopping_patience,
        close_idx=close_idx,
    )

    model.eval()
    all_new_pseudo = []
    with torch.no_grad():
        for batch in unlabeled_loader:
            batch_pseudo = _generate_pseudo_labels_batch(
                model,
                batch,
                device,
                outcome_mode,
                action_threshold,
                outcome_threshold,
                min_rr,
                close_idx,
                seq_len,
            )
            all_new_pseudo.extend(batch_pseudo)

    if not all_new_pseudo:
        print('No new pseudo-labels, stopping self-training.')
        return None
    print(
        f'Generated {len(all_new_pseudo)} pseudo-labels. '
        'Updating labels.parquet...'
    )
    return all_new_pseudo


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
    val_split: float = 0.2,
    log_dir: str | None = None,
    early_stopping_patience: int = 0,
):
    """Run iterative self-training using a separate pool of unlabeled data."""
    # ---------- Initial labeled loader ----------
    loader, df = build_loader_from_parquet(
        features_path, labels_path, order_blocks,
        seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
        batch_size=batch_size, shuffle=True,
    )
    train_loader, val_loader = _split_train_val(loader, val_split, batch_size)
    class_weight = _compute_class_weights(df['action'].to_numpy())

    # ---------- Unlabeled loader ----------
    unlabeled_loader = build_unlabeled_loader_from_parquet(
        features_path_unlabeled, order_blocks,
        seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
        batch_size=batch_size, outcome_mode=outcome_mode,
    )

    # ---------- Self-training rounds ----------
    for round_idx in range(num_rounds):
        pseudo_labels = _self_training_round(
            model, round_idx, num_rounds,
            train_loader, val_loader, unlabeled_loader,
            device, outcome_mode, lambda_outcome, lr,
            epochs_per_round, action_threshold, outcome_threshold,
            min_rr, close_idx, seq_len, class_weight, log_dir,
            save_best=True,
            best_model_path=(
                f'{save_model_path}_best.pt'
                if save_model_path
                else None
            ),
            early_stopping_patience=early_stopping_patience,
        )
        if pseudo_labels is None:
            break

        _update_labels_parquet(labels_path, pseudo_labels, outcome_mode)

        # Rebuild labeled loaders with updated labels
        loader, df = build_loader_from_parquet(
            features_path, labels_path, order_blocks,
            seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
            batch_size=batch_size, shuffle=True,
        )
        train_loader, val_loader = _split_train_val(
            loader, val_split, batch_size
        )

        if save_model_path:
            torch.save(
                model.state_dict(),
                f'{save_model_path}_round{round_idx + 1}.pt',
            )

    return model
