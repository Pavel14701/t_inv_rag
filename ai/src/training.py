"""Training and self-training pipelines for the EntryExitTransformer.

Provides
--------
- :func:`build_loader_from_parquet` - DataLoader for labeled Parquet data.
- :func:`build_unlabeled_loader_from_parquet` - DataLoader for unlabeled
    Parquet data (targets set to ignore).
- :func:`train_one_round` - supervised training loop with validation,
    metrics, checkpointing, TensorBoard logging, and early stopping.
- :func:`self_training_loop` - iterative self-training that pseudo-labels
    an unlabeled pool and adds confident predictions to the training set.

All functions are designed to work with batches of 11 elements (see
:func:`collate_ob` for details).  The training loop supports optional
class weighting, multi-task loss (action + outcome + pattern), and
handles the absence of a validation set gracefully.
"""

from __future__ import annotations

import itertools
import logging
import math

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as functional

from torch.utils.data import DataLoader, Subset
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


logger = logging.getLogger(__name__)


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
    """Create a DataLoader from labeled Parquet files.

    Reads feature and label Parquet files, merges them, stacks the
    columns into a single 2D array, and wraps everything in a
    :class:`TradingDataset`.

    Args:
        features_path: Path to ``features.parquet``.  The file must
            contain at least the columns listed in ``price_cols``,
            ``sig_cols``, and ``tp_sl_cols``.  Indicator columns
            (``ind_cols``) are optional.
        labels_path: Path to ``labels.parquet``.  Must contain the
            columns ``'action'`` (int) and ``'outcome'`` (float).
            If ``pattern_cols`` is given, those columns are expected
            here as well (or in the features file - they will be merged).
        order_blocks: List of all order blocks.  They are filtered
            per window by the dataset.
        seq_len: Number of bars in each sliding window.
        price_cols: Names of the price columns (e.g. OHLCV).
        ind_cols: Names of indicator columns.  Can be empty.
        sig_cols: Names of signal columns (e.g. OB distances).
        tp_sl_cols: Names of the two TP/SL columns.
        batch_size: Batch size for the returned DataLoader.
        shuffle: Whether to shuffle the dataset.
        pattern_cols: Optional list of pattern label column names.
            If provided, they are extracted as a multi-label target
            tensor.

    Returns:
        A tuple ``(loader, merged_df)`` where ``loader`` is a
        PyTorch DataLoader yielding 11-element batches and
        ``merged_df`` is the full joined DataFrame (useful for
        inspecting or computing class weights).

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

    action = df["action"].to_numpy()
    outcome = df["outcome"].to_numpy()

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
    outcome_mode: str = "binary",
    bar_index_col: str | None = "bar_index",
) -> DataLoader:
    """Create a DataLoader for **unlabeled** data.

    The returned loader yields batches where all action targets are
    ``-100`` (ignore) and outcome targets are ``2`` (or ``NaN`` for
    regression).  No pattern targets are included.

    If the features Parquet file contains a column with the name given
    by ``bar_index_col``, its values are passed to
    :class:`TradingDataset` as stable bar identifiers.  This enables
    safe pseudo-label alignment during self-training, even when the
    unlabeled pool differs from the labeled pool.

    Args:
        features_path: Path to the unlabeled features Parquet file.
        order_blocks: List of all order blocks.
        seq_len: Sequence length.
        price_cols: Names of price columns.
        ind_cols: Names of indicator columns (can be empty).
        sig_cols: Names of signal columns.
        tp_sl_cols: Names of TP/SL columns.
        batch_size: Batch size.
        outcome_mode: Determines the ignore value for outcome targets
            (``2`` for binary/multiclass, ``NaN`` for regression).
        bar_index_col: Name of a column that holds a stable bar
            identifier.  If the column exists, its values are used
            for pseudo-label alignment; otherwise positional indices
            are used.

    Returns:
        A DataLoader yielding 11-element batches suitable for
        pseudo-label generation.

    """
    df = load_features_parquet(features_path)
    n = df.height

    action = np.full(n, -100, dtype=int)
    if outcome_mode == "regression":
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
    """Compute inverse-frequency weights for the three action classes.

    Bars with label ``-100`` (ignore) are excluded from the frequency
    calculation.  The returned weights are normalised so that their
    mean is 1.

    Args:
        action_targets: 1D integer array of action labels.

    Returns:
        Float tensor of shape ``(3,)``.

    """
    unique, counts = np.unique(
        action_targets[action_targets != -100], return_counts=True
    )
    weights = np.ones(3, dtype=np.float32)
    for cls, cnt in zip(unique, counts, strict=False):
        if cls in (0, 1, 2):
            weights[int(cls)] = 1.0 / cnt
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _prepare_batch(batch, device: torch.device):
    """Unpack an 11-element batch and move tensors to the given device.

    The batch must be in the format returned by :func:`collate_ob`:
    ``(prices, indicators, signals, tp, sl, order_blocks,
    action_tgt, outcome_tgt, pattern_tgt, start_indices,
    bar_indices)``.

    The ``order_blocks`` element (a list of lists) is **not** moved
    to the device; it stays on the CPU.

    Returns:
        A dictionary with keys matching the batch components.

    """
    (
        prices, indicators, signals, tp, sl,
        order_blocks, action_tgt, outcome_tgt,
        pattern_tgt, start_indices, bar_indices,
    ) = batch
    return {
        "prices": prices.to(device),
        "indicators": indicators.to(device),
        "signals": signals.to(device),
        "tp": tp.to(device),
        "sl": sl.to(device),
        "order_blocks": order_blocks,
        "action_tgt": action_tgt.to(device),
        "outcome_tgt": outcome_tgt.to(device),
        "pattern_tgt": pattern_tgt.to(device),
        "start_indices": start_indices,
        "bar_indices": bar_indices,
    }


def _model_forward_loss(
    model: torch.nn.Module,
    batch_data: dict,
    outcome_mode: str,
    lambda_outcome: float,
    lambda_pattern: float,
    class_weight: torch.Tensor | None,
):
    """Execute a forward pass and compute the combined loss.

    Args:
        model: The transformer model.
        batch_data: Dictionary returned by :func:`_prepare_batch`.
        outcome_mode: Passed to :func:`dual_loss`.
        lambda_outcome: Passed to :func:`dual_loss`.
        lambda_pattern: Passed to :func:`dual_loss`.
        class_weight: Optional class weights for the action head.

    Returns:
        A tuple ``(loss, action_logits, outcome_logits)``.

    """
    action_logits, outcome_logits, pattern_logits = model(
        batch_data["prices"],
        batch_data["indicators"],
        batch_data["signals"],
        batch_data["tp"],
        batch_data["sl"],
        batch_data["order_blocks"],
    )
    loss, _, _, _ = dual_loss(
        action_logits,
        outcome_logits,
        batch_data["action_tgt"],
        batch_data["outcome_tgt"],
        outcome_mode,
        lambda_outcome,
        pattern_logits=pattern_logits,
        pattern_targets=batch_data["pattern_tgt"],
        lambda_pattern=lambda_pattern,
        class_weight=class_weight,
    )
    return loss, action_logits, outcome_logits


def _compute_val_metrics(
    all_action_logits: list[torch.Tensor],
    all_action_targets: list[torch.Tensor],
    all_outcome_targets: list[torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    """Concatenate accumulated tensors and compute action
        accuracy & trade metrics.

    Args:
        all_action_logits: List of flattened ``(N, 3)`` tensors.
        all_action_targets: List of flattened ``(N,)`` long tensors.
        all_outcome_targets: List of flattened ``(N,)`` float tensors.

    Returns:
        A tuple ``(action_acc_dict, trade_metrics_dict)``.

    """
    cat_action_logits = torch.cat(all_action_logits)
    cat_action_targets = torch.cat(all_action_targets)
    action_acc = compute_action_accuracy(cat_action_logits, cat_action_targets)

    cat_outcome_targets = torch.cat(all_outcome_targets)
    trade_metrics = compute_trade_metrics(
        cat_action_logits,
        cat_action_targets,
        cat_outcome_targets,
    )
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
    """Train the model for one epoch.

    Iterates over the DataLoader, calls :func:`_model_forward_loss`,
    and updates parameters.

    Returns:
        Average training loss across all batches.

    """
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
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Validate the model for one epoch.

    Computes loss without gradient and accumulates the logits, action
    targets, and outcome targets needed for metric computation.

    Returns:
        A tuple ``(avg_val_loss, action_acc, trade_metrics)``.

    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    a_logits, a_targets, o_targets = [], [], []

    with torch.no_grad():
        for batch in loader:
            data = _prepare_batch(batch, device)
            loss, action_logits, outcome_logits = _model_forward_loss(
                model, data, outcome_mode, lambda_outcome,
                lambda_pattern, None
            )
            total_loss += loss.item()
            num_batches += 1

            a_logits.append(action_logits.reshape(-1, 3))
            a_targets.append(data["action_tgt"].reshape(-1))
            o_targets.append(data["outcome_tgt"].reshape(-1))

    avg_loss = total_loss / max(num_batches, 1)
    action_acc, trade_metrics = _compute_val_metrics(
        a_logits, a_targets, o_targets
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
    """Print a one-line epoch summary and, if a writer is given,
    log scalars to TensorBoard.

    Scalars logged:
        - ``Loss/train``, ``Loss/val``
        - ``Acc/overall``, ``Acc/entry``
        - ``Trades/win_rate``, ``Trades/profit_factor``
    """
    msg = (
        f'Epoch {epoch + 1}/{total_epochs} | '
        f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
        f'Acc: {action_acc["overall"]:.3f} (entry: {action_acc["entry"]:.3f}) | '  # noqa: E501
        f'WR: {trade_metrics["win_rate"]:.3f} PF: {trade_metrics["profit_factor"]:.2f}'  # noqa: E501
    )
    logger.info(msg)

    if writer is not None:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/overall", action_acc["overall"], epoch)
        writer.add_scalar("Acc/entry", action_acc["entry"], epoch)
        writer.add_scalar("Trades/win_rate", trade_metrics["win_rate"], epoch)
        writer.add_scalar(
            "Trades/profit_factor",
            trade_metrics["profit_factor"],
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
    """Update best loss, save model if improved, and signal early stop.

    Returns:
        A tuple ``(best_val_loss, no_improve_count, should_stop)``.

    """
    if save_best and val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        if best_model_path is not None:
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Best model saved (val_loss=%.4f)", best_val_loss)
    else:
        no_improve_count += 1

    should_stop = (
        early_stopping_patience > 0
        and no_improve_count >= early_stopping_patience
    )
    if should_stop:
        print("  Early stopping.")
    return best_val_loss, no_improve_count, should_stop


def train_one_round(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    device: torch.device,
    outcome_mode: str = "binary",
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
    lambda_pattern: float = 0.1,
    class_weight: torch.Tensor | None = None,
    log_dir: str | None = None,
    save_best: bool = True,
    best_model_path: str | None = None,
    early_stopping_patience: int = 0,
    close_idx: int = 3,  # kept for API compatibility, not used currently
):
    """Run a full supervised training loop.

    The loop performs ``epochs`` passes over ``train_loader``.
    After each training epoch, if ``val_loader`` is not ``None``,
    validation metrics are computed, logged, and used for learning
    rate scheduling and early stopping.  If ``val_loader`` is ``None``,
    only the training loss is reported.

    Args:
        model: The transformer model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.  If ``None``,
            validation, checkpointing, and early stopping are disabled.
        epochs: Total number of epochs.
        device: PyTorch device.
        outcome_mode: One of ``'binary'``, ``'multiclass'``,
            ``'regression'``.
        lambda_outcome: Weight of the outcome loss.
        lr: Initial learning rate for AdamW.
        lambda_pattern: Weight of the pattern loss.
        class_weight: Optional (3,) tensor of class weights for the
            action cross-entropy loss.
        log_dir: If not ``None``, TensorBoard logs are written here.
        save_best: If ``True`` and ``val_loader`` is provided, the
            model with the lowest validation loss is saved.
        best_model_path: File path for the best checkpoint.  Required
            when ``save_best=True``.
        early_stopping_patience: Number of epochs without improvement
            after which training stops.  ``0`` disables early stopping.
        close_idx: **Not used** in the current version; kept for
            API compatibility.

    Returns:
        The trained model (same object as ``model``).

    """
    writer = SummaryWriter(log_dir) if log_dir else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    best_val_loss = math.inf
    no_improve_count = 0

    for epoch in range(epochs):
        train_loss = _run_train_epoch(
            model, train_loader, device, optimizer,
            outcome_mode, lambda_outcome, lambda_pattern, class_weight
        )

        if val_loader is not None:
            val_loss, action_acc, trade_metrics = _run_val_epoch(
                model, val_loader, device,
                outcome_mode, lambda_outcome, lambda_pattern
            )
            _log_epoch(
                writer, epoch, epochs, train_loss, val_loss,
                action_acc, trade_metrics
            )
            scheduler.step(val_loss)

            best_val_loss, no_improve_count, stop = _checkpoint_and_stop(
                val_loss, best_val_loss, no_improve_count, model,
                save_best, best_model_path, early_stopping_patience
            )
            if stop:
                break
        else:
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f",
                epoch + 1,
                epochs,
                train_loss,
            )
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

    if writer:
        writer.close()
    return model


def _check_rr_valid(
    entry_price: float,
    tp_price: float,
    sl_price: float,
    min_rr: float,
) -> bool:
    """Return ``True`` if the risk-reward ratio meets ``min_rr``.

    The direction (long/short) is inferred from the relative positions
    of the entry, TP, and SL levels.
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
    """Convert a raw outcome logit into a pseudo-outcome if confident.

    The confidence threshold is applied symmetrically for binary mode
    (probability > threshold -> win, < 1-threshold -> loss).  For
    multiclass and regression modes the logic is mode-specific.

    Returns:
        A float outcome value, or ``None`` if the model is not
        confident enough.

    """
    if outcome_mode == "binary":
        prob = torch.sigmoid(outcome_logit)
        if prob > outcome_threshold:
            return 1.0
        return 0.0 if prob < (1 - outcome_threshold) else None
    elif outcome_mode == "multiclass":
        probs = functional.softmax(outcome_logit, dim=-1)
        max_prob, cls = probs.max(dim=-1)
        return float(cls.item()) if max_prob >= outcome_threshold else None
    elif outcome_mode == "regression":
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
    """Generate pseudo-labels for one batch of unlabeled data.

    For every bar whose action target is ``-100`` (i.e. unlabeled),
    the model's prediction is evaluated:

    - The predicted action class must have a probability above
        ``action_threshold`` and be either ``entry`` (1) or ``exit`` (2).
    - For ``entry`` predictions, the trade must satisfy the
        risk-reward ratio ``min_rr`` (using the bar's TP/SL levels).
    - The predicted outcome must meet the ``outcome_threshold``
        confidence criterion.

    The stable bar identifier from the batch (``bar_indices[bi] + ti``)
    is used as the global index, ensuring that the pseudo-label can be
    safely written to the original ``labels.parquet`` even when the
    unlabeled pool differs from the labeled pool.

    Args:
        model: Model in evaluation mode.
        batch: 11-element tuple as returned by the DataLoader.
        device: PyTorch device.
        outcome_mode: Prediction mode.
        action_threshold: Minimum probability for a predicted action.
        outcome_threshold: Minimum confidence for the outcome.
        min_rr: Minimum required reward-to-risk ratio for entries.
        close_idx: Index of the close price within the price tensor
            (usually 3 for OHLCV).
        seq_len: Length of the temporal window (T).

    Returns:
        A list of ``(global_bar, pseudo_action, pseudo_outcome)``
        tuples.

    """
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
        global_bar = int(bar_indices[bi].item()) + ti
        new_pseudo.append((global_bar, pseudo_action, pseudo_outcome))

    return new_pseudo


def _update_labels_parquet(
    labels_path: str,
    new_pseudo: list[tuple[int, int, float]],
    outcome_mode: str = "binary",
    round_idx: int = 0,
) -> int:
    """Persist pseudo-labels without destructive mutation (TZ-06 item 2.3).

    Before any modification the current ``labels.parquet`` is backed up
    to ``<path>.bak_round<round_idx>`` so a failed or corrupted round can
    be rolled back with :func:`_rollback_labels`.  A boolean ``is_pseudo``
    column is added on first write to keep human labels and pseudo-labels
    separable; bars already marked ``is_pseudo`` are *never overwritten*
    by a later round (prevents iterative error accumulation).

    Rows that fall outside the length of the file are ignored.

    Args:
        labels_path: Path to ``labels.parquet``.
        new_pseudo: List of ``(global_bar, action, outcome)`` tuples
            as returned by :func:`_generate_pseudo_labels_batch`.
        outcome_mode: Used to choose the fill value for missing
            outcome columns.
        round_idx: Self-training round index (for the backup name).

    Returns:
        Number of pseudo-labels actually applied (skips already-pseudo
        and out-of-range bars).

    """
    import shutil

    df_lbl = load_labels_parquet(labels_path)

    # Per-round backup for rollback.
    backup_path = f"{labels_path}.bak_round{round_idx}.parquet"
    if not Path(backup_path).exists():
        shutil.copyfile(labels_path, backup_path)

    if "action" not in df_lbl.columns:
        df_lbl = df_lbl.with_columns([
            pl.lit(-100).alias("action"),
            pl.lit(
                float("nan") if outcome_mode == "regression" else 2.0
            ).alias("outcome"),
        ])
    if "is_pseudo" not in df_lbl.columns:
        df_lbl = df_lbl.with_columns([pl.lit(False).alias("is_pseudo")])

    action_arr = df_lbl["action"].to_numpy().copy()
    outcome_arr = df_lbl["outcome"].to_numpy().copy()
    is_pseudo_arr = df_lbl["is_pseudo"].to_numpy().copy()
    n_rows = len(action_arr)

    applied = 0
    for global_bar, pseudo_action, pseudo_outcome in new_pseudo:
        if not (0 <= global_bar < n_rows):
            continue
        if bool(is_pseudo_arr[global_bar]):
            # Already a pseudo-label from a previous round: do not
            # re-overwrite it (the model could keep ``correcting`` its
            # own earlier confident answers).
            continue
        action_arr[global_bar] = pseudo_action
        outcome_arr[global_bar] = pseudo_outcome
        is_pseudo_arr[global_bar] = True
        applied += 1

    df_lbl = df_lbl.with_columns([
        pl.Series("action", action_arr),
        pl.Series("outcome", outcome_arr),
        pl.Series("is_pseudo", is_pseudo_arr),
    ])
    save_labels_parquet(df_lbl, labels_path)
    return applied


def _rollback_labels(labels_path: str, round_idx: int) -> None:
    """Restore ``labels.parquet`` from a per-round backup (TZ-06 item 2.3).

    If the backup for ``round_idx`` exists it is copied back over
    ``labels_path``.  Useful for undoing a bad self-training round whose
    pseudo-labels corrupted the dataset.

    Args:
        labels_path: Path to the live ``labels.parquet``.
        round_idx: Round whose backup should be restored.

    Raises:
        FileNotFoundError: If no backup exists for that round.

    """
    import shutil

    backup_path = f"{labels_path}.bak_round{round_idx}.parquet"
    if not Path(backup_path).exists():
        raise FileNotFoundError(
            f"No backup for round {round_idx}: {backup_path}"
        )
    shutil.copyfile(backup_path, labels_path)


def _split_train_val(
    loader: DataLoader,
    val_split: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    """Split a dataset into chronological train/validation loaders.

    The validation set takes the **most recent** ``val_split`` fraction
    of the sliding windows (by time), never a random subset.  To prevent
    leakage across the boundary, training windows are additionally
    truncated so that no training window overlaps any validation bar
    (a window of ``seq_len`` bars would otherwise share
    ``seq_len - 1`` bars with the validation period).

    Args:
        loader: DataLoader whose ``.dataset`` is a :class:`TradingDataset`.
        val_split: Fraction of the most recent windows to use for
            validation.
        batch_size: Batch size for both returned loaders.

    Returns:
        ``(train_loader, val_loader)`` where ``val_loader`` is ``None``
        if ``val_split <= 0``.

    """
    dataset = loader.dataset
    assert isinstance(dataset, TradingDataset), (
        "loader.dataset must be a TradingDataset"
    )
    if val_split <= 0:
        return loader, None

    n_windows = len(dataset)
    n_val = int(n_windows * val_split)
    if n_val == 0:
        raise ValueError("val_split too small, validation set is empty")

    # Validation: the most recent windows
    val_start = n_windows - n_val
    val_indices = list(range(val_start, n_windows))

    # Training: all windows that end strictly before the first
    # validation window starts (no bar overlap across the boundary)
    train_end = max(0, val_start - (dataset.seq_len - 1))
    if train_end == 0:
        raise ValueError(
            "val_split/seq_len leave no training windows: need "
            f"at least {dataset.seq_len} windows before the validation "
            "boundary. Use more data, a smaller seq_len or val_split."
        )
    train_indices = list(range(train_end))

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ob,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
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
    val_loader: DataLoader | None,
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
    """Perform one round of self-training.

    1. Fine-tune the model on the current labeled set.
    2. Run inference on the unlabeled pool and collect pseudo-labels.
    3. Return the list of pseudo-labels (or ``None`` if none were
        generated).

    Returns:
        List of pseudo-labels, or ``None``.

    """
    logger.info(
        "=== Self-training round %d/%d ===", round_idx + 1, num_rounds
    )
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
        logger.info("No new pseudo-labels, stopping self-training.")
        return None
    logger.info(
        "Generated %d pseudo-labels. Updating labels...",
        len(all_new_pseudo),
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
    outcome_mode: str = "binary",
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
    """Run iterative self-training.

    The pipeline alternates between supervised training on the current
    labeled data and pseudo-labeling of a separate unlabeled pool.
    Confident pseudo-labels are merged back into the labels file, and
    the labeled loaders are rebuilt for the next round.

    If ``val_split > 0``, a validation set is carved out of the
    labeled data and used to monitor performance and to keep the best
    model.  When ``val_split <= 0``, no validation is performed.

    Args:
        model: Initialised transformer model.
        features_path: Path to the labeled features Parquet file.
        labels_path: Path to the labels Parquet file (will be updated
            in-place).
        features_path_unlabeled: Path to the unlabeled features
            Parquet file.
        order_blocks: List of all order blocks.
        price_cols: Names of price columns.
        ind_cols: Names of indicator columns.
        sig_cols: Names of signal columns.
        tp_sl_cols: Names of TP/SL columns.
        seq_len: Sequence length.
        batch_size: Batch size for all loaders.
        device: PyTorch device.
        outcome_mode: Outcome prediction mode.
        lambda_outcome: Outcome loss weight.
        lr: Learning rate.
        epochs_per_round: Number of training epochs per round.
        num_rounds: Maximum number of self-training rounds.
        action_threshold: Confidence threshold for action.
        outcome_threshold: Confidence threshold for outcome.
        min_rr: Minimum risk-reward ratio for entry pseudo-labels.
        close_idx: Index of the close price in the price features.
        save_model_path: If set, per-round checkpoints are saved
            with suffix ``_roundN.pt``.  Additionally, the best model
            (if validation is enabled) is saved as ``_best.pt``.
        val_split: Fraction of labeled data used for validation.
            ``0`` disables validation.
        log_dir: TensorBoard log directory (optional).
        early_stopping_patience: Patience for early stopping during
            each training round.

    Returns:
        The trained model.

    """
    # ---------- Initial labeled loader ----------
    loader, df = build_loader_from_parquet(
        features_path, labels_path, order_blocks,
        seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
        batch_size=batch_size, shuffle=True,
    )
    train_loader, val_loader = _split_train_val(loader, val_split, batch_size)
    class_weight = _compute_class_weights(df["action"].to_numpy())

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
                f"{save_model_path}_best.pt"
                if save_model_path
                else None
            ),
            early_stopping_patience=early_stopping_patience,
        )
        if pseudo_labels is None:
            break

        _update_labels_parquet(
            labels_path, pseudo_labels, outcome_mode, round_idx=round_idx
        )
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
                f"{save_model_path}_round{round_idx + 1}.pt",
            )

    return model
