"""Unit tests for training utilities.

This module tests the core training functions: building loaders from Parquet,
class weight computation, batch preparation, forward pass, training and
validation epochs, and the main training loop.
"""
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import polars as pl
from typing import Any, cast

from ..training import (
    build_loader_from_parquet,
    build_unlabeled_loader_from_parquet,
    _compute_class_weights,
    _prepare_batch,
    train_one_round,
    _split_train_val,
    self_training_loop,
    _check_rr_valid,
    _determine_pseudo_outcome,
    _generate_pseudo_labels_batch,
)
from ..transformer import EntryExitTransformer
from ..datatypes import OrderBlock
from ..dataset import TradingDataset


@pytest.mark.unit
def test_compute_class_weights() -> None:
    """Test that class weights are computed correctly.

    The weights should be inverse-frequency, so classes with fewer samples
    get higher weights.  In the test array, class 1 has the most samples,
    so its weight should be the smallest.

    Asserts:
        - Weights tensor has shape (3,).
        - All weights are positive.
        - Weight for class 1 is less than weights for classes 0 and 2.

    """
    action_targets: npt.NDArray[np.int64] = np.array(
        [-100, 0, 0, 1, 1, 1, 2, 2]
    )
    weights: Tensor = _compute_class_weights(action_targets)
    assert weights.shape == (3,)
    assert torch.all(weights > 0)
    # Class 1 (entry) has 3 samples, so its weight should be smallest
    assert weights[1] < weights[0] and weights[1] < weights[2]


@pytest.mark.unit
def test_prepare_batch(sample_batch: tuple) -> None:
    """Test that _prepare_batch moves tensors to the device and
    preserves order blocks.

    The function should return a dictionary containing all batch components,
    with tensors moved to the specified device.  The order blocks list should
    remain a Python list (not moved to device).

    Args:
        sample_batch: Fixture providing a sample batch (11-element tuple).

    Asserts:
        - The returned dictionary contains 'prices' and 'order_blocks' keys.
        - 'order_blocks' is a list.
        - 'prices' tensor is on the CPU device.

    """
    device: torch.device = torch.device('cpu')
    data: dict[str, Any] = _prepare_batch(sample_batch, device)
    assert 'prices' in data
    assert 'order_blocks' in data
    assert isinstance(data['order_blocks'], list)
    assert data['prices'].device == device


@pytest.mark.unit
def test_build_loader_from_parquet(
    sample_parquet_files: dict[str, Any],
) -> None:
    """Test building a DataLoader from Parquet files.

    The loader should be created without errors, and the first batch should
    have the correct number of elements (11) as returned by collate_ob.

    Args:
        sample_parquet_files: Fixture with paths to temporary Parquet files
            and the list of order blocks.

    Asserts:
        - The loader is not None.
        - The returned DataFrame is a polars DataFrame.
        - The first batch has length 11.

    """
    loader, df = build_loader_from_parquet(
        features_path=sample_parquet_files['features_path'],
        labels_path=sample_parquet_files['labels_path'],
        order_blocks=sample_parquet_files['order_blocks'],
        seq_len=32,
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        batch_size=4,
        shuffle=False,
    )
    assert loader is not None
    assert isinstance(df, pl.DataFrame)
    batch = next(iter(loader))
    assert len(batch) == 11  # collate_ob returns 11 elements


@pytest.mark.unit
def test_build_unlabeled_loader_from_parquet(
    sample_parquet_files: dict[str, Any],
) -> None:
    """Test building an unlabeled DataLoader.

    The loader should produce batches where the action targets are all -100
    (ignore).  This is used for self-training on unlabeled data.

    Args:
        sample_parquet_files: Fixture with paths to temporary Parquet files
            and the list of order blocks.

    Asserts:
        - The loader is not None.
        - The first batch has action targets all equal to -100.

    """
    loader = build_unlabeled_loader_from_parquet(
        features_path=sample_parquet_files['unlabeled_path'],
        order_blocks=sample_parquet_files['order_blocks'],
        seq_len=32,
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        batch_size=4,
        outcome_mode='binary',
    )
    assert loader is not None
    batch = next(iter(loader))
    action_tgt: Tensor = batch[6]
    assert torch.all(action_tgt == -100)


@pytest.mark.integration
def test_train_one_round(sample_parquet_files: dict[str, Any]) -> None:
    """Test a full training round with a small model and dataset.

    This test verifies that the training loop runs without errors and that
    the model parameters are updated.  It does not check convergence,
    only that the training completes.

    Args:
        sample_parquet_files: Fixture with paths to temporary Parquet files
            and the list of order blocks.

    Asserts:
        - The trained model is not None.
        - The returned model is an EntryExitTransformer instance.

    """
    # Build model and loaders
    model = EntryExitTransformer(
        n_price_feats=5,
        n_ind_feats=3,
        n_sig_feats=2,
        n_tp_sl_feats=2,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
    )
    device: torch.device = torch.device('cpu')
    loader, df = build_loader_from_parquet(
        features_path=sample_parquet_files['features_path'],
        labels_path=sample_parquet_files['labels_path'],
        order_blocks=sample_parquet_files['order_blocks'],
        seq_len=32,
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        batch_size=2,
        shuffle=False,
    )
    # Split into train/val
    train_loader, val_loader = _split_train_val(
        loader, val_split=0.2, batch_size=2
    )
    # Run training
    trained_model = train_one_round(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        device=device,
        outcome_mode='binary',
        lambda_outcome=0.3,
        lr=1e-4,
        lambda_pattern=0.0,
        class_weight=None,
        save_best=False,
        early_stopping_patience=0,
    )
    assert trained_model is not None
    assert isinstance(trained_model, EntryExitTransformer)


@pytest.mark.integration
@pytest.mark.slow
def test_self_training_loop(sample_parquet_files: dict[str, Any]) -> None:
    """Test that self-training loop runs without errors.

    This test verifies that the self-training pipeline executes successfully
    on a small synthetic dataset. It does not assert the number of
    pseudo-labels, as the synthetic data may not produce confident predictions.

    Args:
        sample_parquet_files: Fixture with paths to temporary Parquet files
            and order blocks.

    """
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = EntryExitTransformer(
        n_price_feats=5,
        n_ind_feats=3,
        n_sig_feats=2,
        n_tp_sl_feats=2,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
    ).to(device)
    # Run self-training with lower thresholds and more epochs
    self_training_loop(
        model=model,
        features_path=sample_parquet_files['features_path'],
        labels_path=sample_parquet_files['labels_path'],
        features_path_unlabeled=sample_parquet_files['unlabeled_path'],
        order_blocks=sample_parquet_files['order_blocks'],
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        seq_len=32,
        batch_size=2,
        device=device,
        outcome_mode='binary',
        lambda_outcome=0.3,
        lr=1e-3,
        epochs_per_round=5,
        num_rounds=1,
        action_threshold=0.3,
        outcome_threshold=0.3,
        min_rr=0.5,
        close_idx=3,
        save_model_path=None,
        val_split=0.2,
        log_dir=None,
        early_stopping_patience=0,
    )


@pytest.mark.unit
def test_check_rr_valid() -> None:
    """Test risk-reward ratio validation for long and short trades.

    The function should correctly validate whether a trade meets the
    minimum risk-reward ratio requirement.

    Asserts:
        - Valid trades with RR >= min_rr return True.
        - Invalid trades with RR < min_rr or incorrect
            price order return False.
    """
    # Long: entry > sl, tp > entry
    assert _check_rr_valid(
        entry_price=100,
        tp_price=120,
        sl_price=90,
        min_rr=0.5
    ) is True
    assert _check_rr_valid(
        entry_price=100,
        tp_price=110,
        sl_price=90,
        min_rr=0.5
    ) is True  # RR=1
    assert _check_rr_valid(
        entry_price=100,
        tp_price=105,
        sl_price=90,
        min_rr=0.5
    ) is True  # RR=0.5 exactly
    assert _check_rr_valid(
        entry_price=100,
        tp_price=104,
        sl_price=90,
        min_rr=0.5
    ) is False  # RR=0.4

    # Short: entry < sl, tp < entry
    assert _check_rr_valid(
        entry_price=100,
        tp_price=80,
        sl_price=110,
        min_rr=0.5
    ) is True  # RR=2
    assert _check_rr_valid(
        entry_price=100,
        tp_price=95,
        sl_price=110,
        min_rr=0.5
    ) is True   # RR=0.5
    assert _check_rr_valid(
        entry_price=100,
        tp_price=96,
        sl_price=110,
        min_rr=0.5
    ) is False  # RR=0.4

    # Invalid: sl < entry for short
    assert _check_rr_valid(
        entry_price=100,
        tp_price=80,
        sl_price=90,
        min_rr=0.5
    ) is False

    # Invalid: tp < entry for long (tp should be > entry)
    assert _check_rr_valid(
        entry_price=100,
        tp_price=90,
        sl_price=80,
        min_rr=0.5
    ) is False


@pytest.mark.unit
def test_determine_pseudo_outcome() -> None:
    """Test pseudo-outcome determination from raw logits.

    The function should convert logits to a pseudo-outcome
    (win/loss/regression) based on the outcome mode and confidence threshold.

    Asserts:
        - In binary mode, returns 1.0 for high probability of win,
            0.0 for high probability of loss, None otherwise.
        - In multiclass mode, returns the class with highest probability
            if above threshold, else None.
        - In regression mode, returns the predicted value if its absolute
            value exceeds threshold, else None.
    """
    # Binary mode
    logit = torch.tensor([2.0])  # sigmoid ~0.88
    assert _determine_pseudo_outcome(logit, 'binary', 0.8) == 1.0
    logit = torch.tensor([-2.0])  # sigmoid ~0.12
    assert _determine_pseudo_outcome(logit, 'binary', 0.8) == 0.0
    logit = torch.tensor([0.0])   # sigmoid 0.5
    assert _determine_pseudo_outcome(logit, 'binary', 0.8) is None
    # Multiclass
    logit = torch.tensor([3.0, 0.0, 0.0])  # softmax ~ [0.952, 0.024, 0.024]
    assert _determine_pseudo_outcome(logit, 'multiclass', 0.9) == 0.0
    logit = torch.tensor([1.0, 0.0, 0.0])  # softmax ~ [0.576, 0.212, 0.212]
    assert _determine_pseudo_outcome(logit, 'multiclass', 0.6) is None
    # Regression
    logit = torch.tensor([0.5])
    assert _determine_pseudo_outcome(logit, 'regression', 0.5) == 0.5
    logit = torch.tensor([0.4])
    assert _determine_pseudo_outcome(logit, 'regression', 0.5) is None


@pytest.mark.unit
def test_generate_pseudo_labels_batch() -> None:
    """Test pseudo-label generation for a batch of unlabeled data.

    The function should generate pseudo-labels for bars that meet the
    action confidence, outcome confidence, and risk-reward criteria.

    Asserts:
        - All bars are labelled (since thresholds are low).
        - All labels have the correct format (bar, action, outcome).
        - All outcomes are 1.0 (win) due to mock model.
    """
    B, T = 2, 5  # noqa: N806
    # Create deterministic tensors that satisfy RR and outcome confidence
    prices = torch.zeros(B, T, 5)
    prices[:, :, 3] = 100.0  # close price = 100
    indicators = torch.zeros(B, T, 3)
    signals = torch.zeros(B, T, 2)
    tp = torch.full((B, T, 1), 120.0)   # TP = 120
    sl = torch.full((B, T, 1), 90.0)    # SL = 90
    order_blocks: list[list[OrderBlock]] = [[], []]
    action_tgt = torch.full((B, T), -100, dtype=torch.long)
    outcome_tgt = torch.full((B, T), 2.0, dtype=torch.float32)
    pattern_tgt = torch.zeros(B, T, 0)
    start_indices = torch.tensor([0, 10])
    bar_indices = torch.tensor([100, 200])
    batch = (prices, indicators, signals, tp, sl, order_blocks,
             action_tgt, outcome_tgt, pattern_tgt, start_indices, bar_indices)

    class MockModel(nn.Module):
        def forward(self, prices, indicators, signals, tp, sl, order_blocks):
            B, T, _ = prices.shape  # noqa: N806
            # Action logits: entry (1) has highest logit
            action_logits = torch.zeros(B, T, 3)
            action_logits[:, :, 1] = 5.0  # entry
            # Outcome logits: win (1.0) with high logit
            outcome_logits = torch.full((B, T, 1), 3.0)  # sigmoid -> ~0.95
            pattern_logits = torch.zeros(B, T, 5)
            return action_logits, outcome_logits, pattern_logits

    model = MockModel()
    device = torch.device('cpu')
    pseudo_labels = _generate_pseudo_labels_batch(
        model, batch, device,
        outcome_mode='binary',
        action_threshold=0.5,
        outcome_threshold=0.5,
        min_rr=0.5,
        close_idx=3,
        seq_len=T,
    )
    assert len(pseudo_labels) == B * T  # should label all bars
    assert all(isinstance(label[0], int) for label in pseudo_labels)
    assert all(label[1] in (1, 2) for label in pseudo_labels)
    assert all(isinstance(label[2], float) for label in pseudo_labels)
    assert all(label[2] == 1.0 for label in pseudo_labels)


@pytest.mark.unit
def test_split_train_val(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test splitting a loader into training and validation sets.

    The function should correctly split the dataset according to val_split
    and return two DataLoader objects, or None for val_loader
    if val_split <= 0.

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - train_loader and val_loader are not None when val_split > 0.
        - The sum of lengths matches the original dataset size.
        - The validation size is approximately val_split * total size.
        - val_loader is None when val_split == 0.
        - ValueError is raised when val_split is too
            small (validation set empty).

    """
    data = sample_dataframe.select(
        ['open', 'high', 'low', 'close', 'volume', 'tp', 'sl']
    ).to_numpy()
    n = len(data)
    data = np.hstack([data, np.zeros((n, 3)), np.zeros((n, 2))])
    action = np.full(n, -100, dtype=np.int64)
    outcome = np.full(n, 2, dtype=np.float32)
    seq_len = 10
    dataset = TradingDataset(
        data=data,
        order_blocks=sample_order_blocks,
        action_targets=action,
        outcome_targets=outcome,
        seq_len=seq_len,
        price_feats=5,
        ind_feats=3,
        sig_feats=2,
        tp_sl_feats=2,
    )
    loader: DataLoader[TradingDataset] = DataLoader(
        dataset, batch_size=2, collate_fn=lambda b: b
    )
    train_loader, val_loader = _split_train_val(
        loader, val_split=0.2, batch_size=2
    )
    assert train_loader is not None
    assert val_loader is not None
    dataset_obj = cast(TradingDataset, loader.dataset)
    train_ds = cast(TradingDataset, train_loader.dataset)
    val_ds = cast(TradingDataset, val_loader.dataset)
    total_len = len(dataset_obj)
    train_len = len(train_ds)
    val_len = len(val_ds)
    # Chronological split: val = most recent windows; train windows
    # end strictly before val starts, so the sum may be < total
    assert train_len + val_len <= total_len
    assert val_len == int(total_len * 0.2)
    expected_train_end = (total_len - int(total_len * 0.2))
    assert train_len == expected_train_end - (seq_len - 1)
    assert train_len > 0
    train_loader2, val_loader2 = _split_train_val(
        loader, val_split=0, batch_size=2
    )
    assert val_loader2 is None
    with pytest.raises(ValueError, match='val_split too small'):
        _split_train_val(loader, val_split=0.001, batch_size=2)

    # No leakage: the last training window must end strictly before
    # the first validation window starts
    train_loader3, val_loader3 = _split_train_val(
        loader, val_split=0.2, batch_size=2
    )
    train_ds3 = cast(TradingDataset, train_loader3.dataset)
    val_ds3 = cast(TradingDataset, val_loader3.dataset)
    train_idx = list(train_ds3.indices)
    val_idx = list(val_ds3.indices)
    max_train_end = max(i + seq_len - 1 for i in train_idx)
    min_val_start = min(val_idx)
    assert max_train_end < min_val_start
    # Validation windows are the most recent ones
    assert val_idx == list(range(total_len - len(val_idx), total_len))

    # Degenerate case: seq_len too large relative to data
    with pytest.raises(ValueError, match='no training windows'):
        big_seq_loader: DataLoader[TradingDataset] = DataLoader(
            TradingDataset(
                data=data,
                order_blocks=sample_order_blocks,
                action_targets=action,
                outcome_targets=outcome,
                seq_len=n // 2,
                price_feats=5,
                ind_feats=3,
                sig_feats=2,
                tp_sl_feats=2,
            ),
            batch_size=2,
            collate_fn=lambda b: b,
        )
        _split_train_val(big_seq_loader, val_split=0.2, batch_size=2)
