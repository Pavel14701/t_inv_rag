"""Quickstart module for the EntryExitTransformer trading system.

Provides a one-function entry point to train the model with minimal
configuration.  All data is assumed to be already prepared as
Parquet files.

Example usage::

    from trading.quickstart import quick_train

    model = quick_train(
        features="data/features.parquet",
        labels="data/labels.parquet",
        order_blocks="data/order_blocks.parquet",
        price_cols=["open", "high", "low", "close", "volume"],
        sig_cols=["dist_supply", "dist_demand"],
        tp_sl_cols=["tp", "sl"],
        epochs=10,
        batch_size=16,
    )
"""

from __future__ import annotations

import torch

from .datatypes import OrderBlock
from .io import load_order_blocks_parquet
from .training import (
    _compute_class_weights,
    _split_train_val,
    build_loader_from_parquet,
    train_one_round,
)
from .transformer import EntryExitTransformer


def quick_train(
    features: str,
    labels: str,
    order_blocks: str,
    price_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    ind_cols: list[str] | None = None,
    pattern_cols: list[str] | None = None,
    seq_len: int = 128,
    batch_size: int = 16,
    epochs: int = 10,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
    hidden_size: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    device: str | None = None,
    val_path: str | None = None,
    val_split: float = 0.2,
    class_weight: bool = True,
    log_dir: str | None = None,
    early_stopping_patience: int = 3,
    save_best_path: str | None = None,
    n_patterns: int = 10,
    **model_kwargs,
) -> EntryExitTransformer:
    """Train the Entry-Exit transformer in a single call.

    All data is read from Parquet files that must exist and be properly
    formatted.  The function creates a model, builds a DataLoader
    (optionally with a validation split), trains for the given number of
    epochs, and returns the trained model.

    Args:
        features: Path to ``features.parquet``.  Must contain at least
            the columns listed in ``price_cols``, ``sig_cols``,
            ``tp_sl_cols``, and optionally ``ind_cols``.
        labels: Path to ``labels.parquet``.  Must contain columns
            ``action`` and ``outcome``.  If ``pattern_cols`` is given,
            those columns are expected here as well (or in the features
            file - they will be merged).
        order_blocks: Path to ``order_blocks.parquet``.
        price_cols: List of price column names (e.g. OHLCV).
        sig_cols: List of signal column names (e.g. distances).
        tp_sl_cols: List of two column names for TP and SL absolute
            prices.
        ind_cols: Optional list of indicator column names.
        pattern_cols: Optional list of pattern label column names.
        seq_len: Length of sliding windows (default 128).
        batch_size: Batch size (default 16).
        epochs: Number of training epochs (default 10).
        outcome_mode: One of 'binary', 'multiclass', 'regression'
            (default 'binary').
        lambda_outcome: Weight of outcome loss (default 0.3).
        lr: Learning rate (default 1e-4).
        hidden_size: Transformer hidden size (default 128).
        num_layers: Number of transformer layers (default 4).
        num_heads: Number of attention heads (default 8).
        device: Torch device string (e.g. 'cuda', 'cpu').  Auto-detected
            if not provided.
        val_path: Optional path to a separate validation Parquet file.
            If not provided, a random split of the training data is used
            (size controlled by ``val_split``).
        val_split: Fraction of training data to use for validation when
            ``val_path`` is not specified (default 0.2).
        class_weight: If True, compute inverse-frequency class weights
            for the action loss (default True).
        log_dir: If set, TensorBoard logs are written there.
        early_stopping_patience: Stop after this many epochs without
            improvement (default 3). 0 disables.
        save_best_path: If set, the model with the lowest validation loss
            is saved to this path.
        n_patterns: Number of pattern labels for the multi-label pattern
            head. Must match the number of columns in ``pattern_cols``
            (default 10). If pattern_cols is None, this value is still
            used to initialise the model but pattern loss is not applied.
        **model_kwargs: Additional keyword arguments forwarded to
            :class:`EntryExitTransformer` constructor.

    Returns:
        Trained :class:`EntryExitTransformer` model.

    """
    # ---------- Device ----------
    if device is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device
    torch_device = torch.device(device_str)
    # ---------- Load order blocks ----------
    obs: list[OrderBlock] = load_order_blocks_parquet(order_blocks)
    # ---------- Build model ----------
    model = EntryExitTransformer(
        n_price_feats=len(price_cols),
        n_ind_feats=len(ind_cols) if ind_cols else 0,
        n_sig_feats=len(sig_cols),
        n_tp_sl_feats=len(tp_sl_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        outcome_mode=outcome_mode,
        n_patterns=n_patterns,
        **model_kwargs,
    ).to(torch_device)
    # ---------- Build labeled loader ----------
    train_loader_all, df = build_loader_from_parquet(
        features_path=features,
        labels_path=labels,
        order_blocks=obs,
        seq_len=seq_len,
        price_cols=price_cols,
        ind_cols=ind_cols or [],
        sig_cols=sig_cols,
        tp_sl_cols=tp_sl_cols,
        batch_size=batch_size,
        shuffle=True,
        pattern_cols=pattern_cols,
    )
    # ---------- Validation split ----------
    if val_path:
        # Separate validation file provided
        val_loader, _ = build_loader_from_parquet(
            features_path=val_path,
            labels_path=labels,  # reuse same labels? Typically labels
                                 # are per-bar and should be matched;
                                 # here we assume the same labels file
                                 # works for the validation features.
            order_blocks=obs,
            seq_len=seq_len,
            price_cols=price_cols,
            ind_cols=ind_cols or [],
            sig_cols=sig_cols,
            tp_sl_cols=tp_sl_cols,
            batch_size=batch_size,
            shuffle=False,
            pattern_cols=pattern_cols,
        )
        train_loader = train_loader_all
    else:
        # Random split from training data
        train_loader, val_loader = _split_train_val(  # type: ignore[assignment]  # noqa: E501
            train_loader_all, val_split, batch_size
        )
    # ---------- Class weights (optional) ----------
    cw = (
        _compute_class_weights(df['action'].to_numpy())
        if class_weight else None
    )
    # ---------- Train ----------
    model = train_one_round(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=torch_device,
        outcome_mode=outcome_mode,
        lambda_outcome=lambda_outcome,
        lr=lr,
        lambda_pattern=0.1 if pattern_cols else 0.0,
        class_weight=cw,
        log_dir=log_dir,
        save_best=True,
        best_model_path=save_best_path,
        early_stopping_patience=early_stopping_patience,
        close_idx=3,  # default OHLCV
    )
    # Ensure the returned module is indeed an EntryExitTransformer
    assert isinstance(model, EntryExitTransformer), (
        'train_one_round returned an unexpected type'
    )
    print('Training finished.')
    return model
