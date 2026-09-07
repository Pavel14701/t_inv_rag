"""Integration tests for the quickstart module.

These tests verify the end-to-end training pipeline using the `quick_train`
function.  They run on small synthetic data to ensure the model builds,
trains, and returns an EntryExitTransformer without errors.
"""

import pytest
import polars as pl
import numpy as np

from ..quickstart import quick_train
from ..transformer import EntryExitTransformer


@pytest.mark.integration
def test_quick_train_basic(sample_parquet_files: dict[str, str]) -> None:
    """Test quick_train with minimal parameters on CPU.

    The function should build a model, train for one epoch, and return
    a trained EntryExitTransformer instance without raising exceptions.

    Args:
        sample_parquet_files: Fixture providing paths to temporary Parquet
            files for features, labels, and order blocks.

    Asserts:
        - The returned object is an instance of EntryExitTransformer.

    """
    model: EntryExitTransformer = quick_train(
        features=sample_parquet_files['features_path'],
        labels=sample_parquet_files['labels_path'],
        order_blocks=sample_parquet_files['order_blocks_path'],
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        seq_len=32,
        batch_size=2,
        epochs=1,
        device='cpu',
    )
    assert isinstance(model, EntryExitTransformer)


@pytest.mark.integration
def test_quick_train_with_validation_file(
    sample_parquet_files: dict[str, str]
) -> None:
    """Test quick_train with a separate validation Parquet file.

    The validation file can be the same as the features file for testing
    purposes.  The model should still train without errors.

    Args:
        sample_parquet_files: Fixture providing paths to temporary
        Parquet files.

    Asserts:
        - The returned object is an instance of EntryExitTransformer.

    """
    model: EntryExitTransformer = quick_train(
        features=sample_parquet_files['features_path'],
        labels=sample_parquet_files['labels_path'],
        order_blocks=sample_parquet_files['order_blocks_path'],
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        seq_len=32,
        batch_size=2,
        epochs=1,
        # same files for simplicity
        val_path=sample_parquet_files['features_path'],
        val_labels_path=sample_parquet_files['labels_path'],
        device='cpu',
        save_best_path=None,
    )
    assert isinstance(model, EntryExitTransformer)


@pytest.mark.integration
def test_quick_train_with_pattern_cols(
    sample_parquet_files: dict[str, str]
) -> None:
    """Test quick_train with pattern columns and corresponding n_patterns.

    Pattern columns are added to the labels file, and `n_patterns` is set
    to match their count.  The model should handle the multi-label pattern
    head correctly.

    Args:
        sample_parquet_files: Fixture providing paths to temporary
        Parquet files.

    Asserts:
        - The returned object is an instance of EntryExitTransformer.

    """
    labels_path: str = sample_parquet_files['labels_path']
    df_lbl = pl.read_parquet(labels_path)
    n_rows: int = len(df_lbl)
    df_lbl = df_lbl.with_columns([
        pl.Series('pattern1', np.random.randint(0, 2, n_rows)),
        pl.Series('pattern2', np.random.randint(0, 2, n_rows)),
    ])
    df_lbl.write_parquet(labels_path)
    model: EntryExitTransformer = quick_train(
        features=sample_parquet_files['features_path'],
        labels=labels_path,
        order_blocks=sample_parquet_files['order_blocks_path'],
        price_cols=['open', 'high', 'low', 'close', 'volume'],
        sig_cols=['sig1', 'sig2'],
        tp_sl_cols=['tp', 'sl'],
        ind_cols=['ind1', 'ind2', 'ind3'],
        pattern_cols=['pattern1', 'pattern2'],
        n_patterns=2,
        seq_len=32,
        batch_size=2,
        epochs=1,
        device='cpu',
        save_best_path=None,
    )
    assert isinstance(model, EntryExitTransformer)
