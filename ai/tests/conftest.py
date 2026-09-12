"""Pytest fixtures for the EntryExitTransformer test suite."""

import tempfile

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
import torch

from ai.src.datatypes import OrderBlock
from ai.src.features import compute_atr


Batch = tuple[
    torch.Tensor,  # prices
    torch.Tensor,  # indicators
    torch.Tensor,  # signals
    torch.Tensor,  # tp
    torch.Tensor,  # sl
    list[list[OrderBlock]],  # order_blocks
    torch.Tensor,  # action_targets
    torch.Tensor,  # outcome_targets
    torch.Tensor,  # pattern_targets
    torch.Tensor,  # start_indices
    torch.Tensor,  # bar_indices
]


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a synthetic DataFrame with OHLCV and other columns."""
    n = 200
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 1.5
    low = close - np.random.rand(n) * 1.5
    open_ = close - np.random.rand(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    tp = close + np.abs(np.random.rand(n) * 2) + 1.0
    sl = np.maximum(close - np.abs(np.random.rand(n) * 2) - 1.0, 1.0)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "tp": tp,
            "sl": sl,
            "bar_index": np.arange(n),
        }
    )


@pytest.fixture
def sample_order_blocks() -> list[OrderBlock]:
    """Create a list of synthetic order blocks."""
    obs = []
    for i in range(5):
        start = i * 40 + 10
        end = start + 15
        zone_low = 95 + i * 2
        zone_high = zone_low + 2
        ob = OrderBlock(
            id=i,
            block_type="demand" if i % 2 == 0 else "supply",
            start=datetime.now(),
            break_=datetime.now(),
            retest=datetime.now(),
            zone_low=zone_low,
            zone_high=zone_high,
            strength=1.0 + i * 0.5,
            structure_label="valid" if i % 2 == 0 else None,
            trend_direction="up" if i < 3 else "down",
            start_idx=start,
            end_idx=end,
        )
        obs.append(ob)
    return obs


@pytest.fixture
def sample_atr(sample_dataframe: pl.DataFrame) -> np.ndarray:
    """Compute ATR for the sample dataframe."""
    return compute_atr(sample_dataframe, period=14)


@pytest.fixture
def sample_batch(
    sample_dataframe: pl.DataFrame, sample_order_blocks: list[OrderBlock]
) -> Batch:
    """Create a small batch (B=2, T=128) for model testing."""
    from ai.src.dataset import TradingDataset

    data = sample_dataframe.select(
        ["open", "high", "low", "close", "volume", "tp", "sl"]
    ).to_numpy()
    n = len(data)
    # Add dummy indicator and signal columns (zeros)
    # ind_feats=3, sig_feats=2
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
        pattern_targets=None,
        bar_index=np.arange(n),
    )
    # Convert to a list of samples and collate manually to avoid dependency
    samples = [dataset[i] for i in range(2)]
    from ai.src.dataset import collate_ob

    return collate_ob(samples)


@pytest.fixture
def sample_action_outcome_labels(
    sample_dataframe: pl.DataFrame,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Return action and outcome arrays from
    generate_labels_from_strategy (or dummy).
    """
    # For testing, we can provide dummy labels.
    n = len(sample_dataframe)
    action = np.random.choice(
        [-100, 0, 1, 2], size=n, p=[0.8, 0.1, 0.05, 0.05]
    )
    outcome = np.full(n, np.nan, dtype=np.float32)
    # Set some outcomes for entry bars
    entry_mask = action == 1
    outcome[entry_mask] = np.random.choice(
        [0.0, 1.0, 2.0], size=entry_mask.sum()
    )
    return action, outcome


@pytest.fixture
def sample_parquet_files(sample_dataframe, sample_order_blocks):
    """Create temporary Parquet files for features,
    labels, and order blocks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = Path(tmpdir) / "features.parquet"
        labels_path = Path(tmpdir) / "labels.parquet"
        order_blocks_path = Path(tmpdir) / "order_blocks.parquet"
        unlabeled_path = Path(tmpdir) / "unlabeled.parquet"
        # Features with all required columns
        df_feat = sample_dataframe.select(
            ["open", "high", "low", "close", "volume", "tp", "sl", "bar_index"]
        )
        n = len(df_feat)
        df_feat = df_feat.with_columns(
            [
                pl.Series("ind1", np.zeros(n)),
                pl.Series("ind2", np.zeros(n)),
                pl.Series("ind3", np.zeros(n)),
                pl.Series("sig1", np.zeros(n)),
                pl.Series("sig2", np.zeros(n)),
            ]
        )
        df_feat.write_parquet(features_path)
        # Labels
        df_lbl = sample_dataframe.select(["bar_index"]).with_columns(
            [
                pl.Series("action", np.full(n, -100, dtype=np.int64)),
                pl.Series("outcome", np.full(n, 2.0, dtype=np.float32)),
            ]
        )
        df_lbl.write_parquet(labels_path)
        # Order blocks
        ob_data = []
        ob_data.extend(
            {
                "id": ob.id,
                "block_type": ob.block_type,
                "start": ob.start,
                "break_": ob.break_,
                "retest": ob.retest,
                "zone_low": ob.zone_low,
                "zone_high": ob.zone_high,
                "strength": ob.strength,
                "structure_label": ob.structure_label,
                "trend_direction": ob.trend_direction,
                "start_idx": ob.start_idx,
                "end_idx": ob.end_idx,
            }
            for ob in sample_order_blocks
        )
        pl.DataFrame(ob_data).write_parquet(order_blocks_path)
        # Unlabeled features (same as features for test)
        df_feat.write_parquet(unlabeled_path)
        yield {
            "features_path": str(features_path),
            "labels_path": str(labels_path),
            "order_blocks_path": str(order_blocks_path),
            "unlabeled_path": str(unlabeled_path),
            "order_blocks": sample_order_blocks,
        }


@pytest.fixture
def large_dataframe() -> pl.DataFrame:
    """Large synthetic DataFrame for slow integration tests."""
    n = 10000
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 1.5
    low = close - np.random.rand(n) * 1.5
    open_ = close - np.random.rand(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    tp = close + np.abs(np.random.rand(n) * 2) + 1.0
    sl = np.maximum(close - np.abs(np.random.rand(n) * 2) - 1.0, 1.0)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "tp": tp,
            "sl": sl,
            "bar_index": np.arange(n),
        }
    )


@pytest.fixture
def large_order_blocks() -> list[OrderBlock]:
    """Large list of order blocks for slow tests."""
    obs = []
    for i in range(100):
        start = i * 80 + 10
        end = start + 30
        zone_low = 95 + i * 0.5
        zone_high = zone_low + 2
        ob = OrderBlock(
            id=i,
            block_type="demand" if i % 2 == 0 else "supply",
            start=datetime.now(),
            break_=datetime.now(),
            retest=datetime.now(),
            zone_low=zone_low,
            zone_high=zone_high,
            strength=1.0 + i * 0.1,
            structure_label="valid" if i % 2 == 0 else None,
            trend_direction="up" if i < 50 else "down",
            start_idx=start,
            end_idx=end,
        )
        obs.append(ob)
    return obs


@pytest.fixture
def model_params() -> dict[str, Any]:
    """Default model parameters for testing.

    Returns:
        Dictionary of model hyperparameters used across tests.

    """
    return {
        "n_price_feats": 5,
        "n_ind_feats": 3,
        "n_sig_feats": 2,
        "n_tp_sl_feats": 2,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "max_seq_len": 128,
        "max_ob_seq_len": 32,
        "n_action_classes": 3,
        "outcome_mode": "binary",
        "n_outcome_classes": 2,
        "n_patterns": 5,
        "ob_embedding_dim": 8,
        "atr_global": 1.0,
    }


@pytest.fixture
def sample_batch_tensors(sample_batch: tuple) -> dict[str, Any]:
    """Extract tensors from a sample batch (B=2, T=128).

    Args:
        sample_batch: Fixture providing a batch of 11 elements.

    Returns:
        Dictionary with keys 'prices', 'indicators', 'signals', 'tp', 'sl',
        and 'order_blocks'.

    """
    batch = sample_batch
    return {
        "prices": batch[0],  # (2, 128, 5)
        "indicators": batch[1],  # (2, 128, 3)
        "signals": batch[2],  # (2, 128, 2)
        "tp": batch[3],  # (2, 128, 1)
        "sl": batch[4],  # (2, 128, 1)
        "order_blocks": batch[5],  # list of 2 lists
    }
