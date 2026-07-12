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
from .training import build_loader_from_parquet, train_one_round
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
    **model_kwargs,
) -> EntryExitTransformer:
    """Train the Entry‑Exit transformer in a single call.

    All data is read from Parquet files that must exist and be properly
    formatted.  The function creates a model, builds a DataLoader,
    trains for the given number of epochs, and returns the trained model.

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
        **model_kwargs: Additional keyword arguments forwarded to
            :class:`EntryExitTransformer` constructor.

    Returns:
        Trained :class:`EntryExitTransformer` model.

    """
    # ---------- Device resolution ----------
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
        **model_kwargs,
    ).to(torch_device)
    # ---------- Build loader ----------
    train_loader, _ = build_loader_from_parquet(
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
    # For simplicity we use the same loader for validation;
    # in practice you'd want a separate validation file.
    val_loader = train_loader
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
    )
    # Ensure model is the correct type (satisfy type checker)
    assert isinstance(model, EntryExitTransformer), (
        'train_one_round returned unexpected type'
    )
    print('Training finished.')
    return model
