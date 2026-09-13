"""Unit tests for I/O functions.

This module tests reading and writing of Parquet files for features, labels,
and order blocks, as well as the merge function for features and labels.
"""

import tempfile

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai.src.datatypes import OrderBlock
from ai.src.io import (
    load_features_parquet,
    load_labels_parquet,
    load_order_blocks_parquet,
    merge_features_labels,
    save_labels_parquet,
)


@pytest.mark.unit
def test_load_features_parquet(sample_dataframe: pl.DataFrame) -> None:
    """Test that features can be loaded from a Parquet file.

    The loaded DataFrame should have the same shape and
    columns as the original.

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.

    Asserts:
        - Loaded DataFrame has same shape as original.
        - Column names match.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "features.parquet"
        sample_dataframe.write_parquet(tmp_path)
        df_loaded = load_features_parquet(str(tmp_path))
        assert df_loaded.shape == sample_dataframe.shape
        assert df_loaded.columns == sample_dataframe.columns


@pytest.mark.unit
def test_load_labels_parquet(sample_dataframe: pl.DataFrame) -> None:
    """Test that labels can be loaded from a Parquet file.

    The loaded DataFrame should contain the required
    'action' and 'outcome' columns.

    Args:
        sample_dataframe: Fixture with bar_index column.

    Asserts:
        - Loaded DataFrame has same shape as original.
        - 'action' and 'outcome' columns are present.

    """
    df_labels = pl.DataFrame(
        {
            "bar_index": sample_dataframe["bar_index"],
            "action": np.random.choice(
                [-100, 0, 1, 2], size=len(sample_dataframe)
            ),
            "outcome": np.random.randn(len(sample_dataframe)),
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "labels.parquet"
        df_labels.write_parquet(tmp_path)
        df_loaded = load_labels_parquet(str(tmp_path))
        assert df_loaded.shape == df_labels.shape
        assert "action" in df_loaded.columns
        assert "outcome" in df_loaded.columns


@pytest.mark.unit
def test_save_labels_parquet(sample_dataframe: pl.DataFrame) -> None:
    """Test that labels can be saved to a Parquet file and reloaded correctly.

    Args:
        sample_dataframe: Fixture with bar_index column.

    Asserts:
        - Loaded DataFrame has same shape as original.
        - All values in 'action' and 'outcome' columns match.

    """
    df_labels = pl.DataFrame(
        {
            "bar_index": sample_dataframe["bar_index"],
            "action": np.random.choice(
                [-100, 0, 1, 2], size=len(sample_dataframe)
            ),
            "outcome": np.random.randn(len(sample_dataframe)),
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "labels.parquet"
        save_labels_parquet(df_labels, str(tmp_path))
        df_loaded = pl.read_parquet(str(tmp_path))
        assert df_loaded.shape == df_labels.shape
        assert (df_loaded["action"] == df_labels["action"]).all()
        assert (df_loaded["outcome"] == df_labels["outcome"]).all()


@pytest.mark.unit
def test_load_order_blocks_parquet(
    sample_order_blocks: list[OrderBlock],
) -> None:
    """Test that order blocks can be saved to and loaded from a Parquet file.

    All fields of the OrderBlock dataclass should be preserved exactly.

    Args:
        sample_order_blocks: Fixture with a list of OrderBlock objects.

    Asserts:
        - Number of loaded blocks matches original.
        - All fields of each block match the original.

    """
    # Convert OrderBlock list to DataFrame
    df_obs = pl.DataFrame(
        [
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
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "order_blocks.parquet"
        df_obs.write_parquet(tmp_path)
        loaded_obs = load_order_blocks_parquet(str(tmp_path))
        assert len(loaded_obs) == len(sample_order_blocks)
        assert all(
            loaded_obs[i].id == sample_order_blocks[i].id
            and loaded_obs[i].block_type == sample_order_blocks[i].block_type
            and loaded_obs[i].zone_low == sample_order_blocks[i].zone_low
            and loaded_obs[i].zone_high == sample_order_blocks[i].zone_high
            and loaded_obs[i].strength == sample_order_blocks[i].strength
            and loaded_obs[i].structure_label
            == sample_order_blocks[i].structure_label
            and loaded_obs[i].trend_direction
            == sample_order_blocks[i].trend_direction
            and loaded_obs[i].start_idx == sample_order_blocks[i].start_idx
            and loaded_obs[i].end_idx == sample_order_blocks[i].end_idx
            for i in range(len(sample_order_blocks))
        )


@pytest.mark.unit
def test_merge_features_labels(sample_dataframe: pl.DataFrame) -> None:
    """Test merging feature and label DataFrames.

    The merge should combine the DataFrames on 'bar_index' if present,
    and ensure that 'action' and 'outcome' columns exist with correct defaults.

    Args:
        sample_dataframe: Fixture with price and TP/SL columns.

    Asserts:
        - Merged DataFrame contains 'action' and 'outcome' columns.
        - Number of rows matches the feature DataFrame.
        - Missing values are filled with defaults.

    """
    df_feat = sample_dataframe.select(
        ["open", "high", "low", "close", "volume", "bar_index"]
    )
    df_lbl = sample_dataframe.select(["bar_index"]).with_columns(
        [
            pl.Series(
                "action",
                np.random.choice([-100, 0, 1, 2], len(sample_dataframe)),
            ),
            pl.Series("outcome", np.random.randn(len(sample_dataframe))),
        ]
    )
    df_merged = merge_features_labels(df_feat, df_lbl)
    assert "action" in df_merged.columns
    assert "outcome" in df_merged.columns
    assert len(df_merged) == len(df_feat)
