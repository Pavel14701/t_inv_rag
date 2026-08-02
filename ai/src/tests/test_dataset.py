"""Unit tests for TradingDataset and collate_ob.

These tests verify the dataset class and collation function used in the
EntryExitTransformer pipeline.
"""

import pytest
import numpy as np
import torch
from torch import Tensor

import polars as pl

from ..dataset import TradingDataset, collate_ob
from ..datatypes import OrderBlock


@pytest.mark.unit
def test_trading_dataset_len(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test that the dataset length equals total rows minus sequence length.

    The dataset should produce one sample for each possible starting position
    of a sliding window of length `seq_len`.  Dummy indicator and signal
    features are added to match the expected feature dimensions.

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - Dataset length equals `n_rows - seq_len`.

    """
    data = sample_dataframe.select(
        ['open', 'high', 'low', 'close', 'volume', 'tp', 'sl']
    ).to_numpy()
    n = len(data)
    # Add dummy indicator (3) and signal (2) features
    data = np.hstack([data, np.zeros((n, 3)), np.zeros((n, 2))])
    action = np.full(n, -100, dtype=np.int64)
    outcome = np.full(n, 2, dtype=np.float32)
    seq_len = 128
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
    expected_len = n - seq_len
    assert len(dataset) == expected_len


@pytest.mark.unit
def test_trading_dataset_getitem(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test that a single sample from the dataset has the
    correct shape and types.

    The item should contain 11 elements: prices, indicators, signals, TP/SL,
    order blocks, action/outcome targets, pattern targets (empty), start index,
    and stable bar index.  The order blocks are filtered to include only those
    whose end index falls within the window.

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - The sample tuple has length 11.
        - The price tensor has shape (seq_len, 5) and dtype float32.
        - All order blocks in the window have valid start/end indices.
        - Action targets have dtype int64 and shape (seq_len,).
        - Outcome targets have dtype float32 and shape (seq_len,).

    """
    data = sample_dataframe.select(
        ['open', 'high', 'low', 'close', 'volume', 'tp', 'sl']
    ).to_numpy()
    n = len(data)
    data = np.hstack([data, np.zeros((n, 3)), np.zeros((n, 2))])
    action = np.full(n, -100, dtype=np.int64)
    outcome = np.full(n, 2, dtype=np.float32)
    seq_len = 128
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
        bar_index=np.arange(n),
    )
    item = dataset[0]
    # There are 11 elements: prices, indicators, signals, tp, sl, ob_window,
    # action_target, outcome_target, pattern_target, start_bar, bar_idx
    assert len(item) == 11
    prices: Tensor = item[0]
    assert prices.shape == (seq_len, 5)
    assert prices.dtype == torch.float32
    # Check order blocks filtering
    ob_window: list[OrderBlock] = item[5]
    assert all(ob.start_idx >= 0 and ob.end_idx < seq_len for ob in ob_window)
    # Check action and outcome targets
    action_tgt: Tensor = item[6]
    assert action_tgt.shape == (seq_len,)
    assert action_tgt.dtype == torch.int64
    outcome_tgt: Tensor = item[7]
    assert outcome_tgt.shape == (seq_len,)
    assert outcome_tgt.dtype == torch.float32


@pytest.mark.unit
def test_collate_ob(
    sample_dataframe: pl.DataFrame,
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test that collate_ob correctly stacks tensors and collects order blocks.

    Given a list of samples (each a tuple of 11 elements), the collation
    function stacks all tensor elements and preserves the list of order
    blocks for each sample.  The output batch should contain 11 elements,
    with the first five being tensors of shape (batch_size, seq_len, ...),
    the sixth being a list of order block lists, and the remaining tensors
    of shape (batch_size, seq_len) or (batch_size,).

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - The batch tuple has length 11.
        - The price tensor has shape (2, seq_len, 5).
        - The order_blocks element is a list of lists of length 2.
        - start_indices and bar_indices are tensors of shape (2,).

    """
    data = sample_dataframe.select(
        ['open', 'high', 'low', 'close', 'volume', 'tp', 'sl']
    ).to_numpy()
    n = len(data)
    data = np.hstack([data, np.zeros((n, 3)), np.zeros((n, 2))])
    action = np.full(n, -100, dtype=np.int64)
    outcome = np.full(n, 2, dtype=np.float32)
    seq_len = 128
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
        bar_index=np.arange(n),
    )
    # Get two samples
    samples = [dataset[0], dataset[1]]
    batch = collate_ob(samples)
    # Batch should contain 11 tensors/lists
    assert len(batch) == 11
    # Check shapes
    prices: Tensor = batch[0]
    assert prices.shape == (2, seq_len, 5)
    # order_blocks is list of lists
    ob_lists: list[list[OrderBlock]] = batch[5]
    assert isinstance(ob_lists, list)
    assert len(ob_lists) == 2
    # start_indices and bar_indices
    start_indices: Tensor = batch[9]
    assert start_indices.shape == (2,)
    bar_indices: Tensor = batch[10]
    assert bar_indices.shape == (2,)
