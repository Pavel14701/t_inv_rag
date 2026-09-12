"""Tests for TZ-06 п.2.2 (chronological split) and п.2.3 (self-training
label isolation via is_pseudo / per-round backup / rollback).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from torch.utils.data import DataLoader

from ai.src.dataset import TradingDataset, collate_ob
from ai.src.io import save_labels_parquet
from ai.src.training import (
    _rollback_labels,
    _split_train_val,
    _update_labels_parquet,
)


def _make_dataset(num_bars: int, seq_len: int) -> TradingDataset:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(
        (num_bars, 12)
    )  # 5 price + 3 ind + 2 sig + 1 tp + 1 sl
    action = np.full(num_bars, -100, dtype=np.int64)
    outcome = np.zeros(num_bars, dtype=np.float32)
    return TradingDataset(
        data,
        [],
        action,
        outcome,
        seq_len=seq_len,
        price_feats=5,
        ind_feats=3,
        sig_feats=2,
        tp_sl_feats=2,
    )


def test_split_train_val_no_window_overlap():
    """п.2.2: no training window shares bars with the validation period."""
    # The validation set is the most recent windows; training windows are
    # truncated before the boundary.
    seq_len = 8
    dataset = _make_dataset(num_bars=200, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_ob)
    train, val = _split_train_val(loader, val_split=0.2, batch_size=8)
    assert val is not None

    train_ends = max(i + seq_len - 1 for i in train.dataset.indices)
    val_starts = min(j for j in val.dataset.indices)
    assert train_ends < val_starts


def test_split_train_val_uses_recent_windows():
    """п.2.2: validation windows come from the later half of the bars."""
    dataset = _make_dataset(num_bars=300, seq_len=4)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_ob)
    train, val = _split_train_val(loader, val_split=0.5, batch_size=8)
    assert val is not None
    # validation windows start in the later half of the bar range
    max_train = max(train.dataset.indices)
    min_val = min(val.dataset.indices)
    assert min_val > max_train


def test_update_labels_adds_is_pseudo_and_backup(tmp_path):
    """п.2.3: first pseudo write adds is_pseudo column and a backup."""
    # п.2.3: first pseudo-label write must add an is_pseudo column and
    # create a per-round backup.
    labels = pl.DataFrame(
        {
            "action": [-100] * 6 + [1] * 4,
            "outcome": [2.0] * 6 + [0.5, 1.5, 1.0, 0.0],
        }
    )
    path = tmp_path / "labels.parquet"
    save_labels_parquet(labels, path)

    applied = _update_labels_parquet(
        str(path),
        [(0, 1, 1.0), (1, 2, 0.0)],
        outcome_mode="binary",
        round_idx=0,
    )
    assert applied == 2

    df = pl.read_parquet(path)
    assert "is_pseudo" in df.columns
    assert df["is_pseudo"].to_list()[:2] == [True, True]
    assert df["is_pseudo"].to_list()[2:] == [False] * 8
    assert (path.with_name("labels.parquet.bak_round0.parquet")).exists()


def test_update_labels_never_overwrites_pseudo(tmp_path):
    """п.2.3: a later round must not overwrite an already-pseudo bar."""
    # only trusting the model's own iterative corrections would let errors
    # accumulate; an is_pseudo bar is frozen.
    labels = pl.DataFrame(
        {
            "action": [-100] * 5,
            "outcome": [2.0] * 5,
        }
    )
    path = tmp_path / "labels.parquet"
    save_labels_parquet(labels, path)

    _update_labels_parquet(
        str(path), [(0, 1, 1.0)], outcome_mode="binary", round_idx=0
    )
    # Second round tries to change bar 0; must be a no-op.
    applied = _update_labels_parquet(
        str(path), [(0, 2, 0.0)], outcome_mode="binary", round_idx=1
    )
    assert applied == 0

    df = pl.read_parquet(path)
    assert df["action"][0] == 1  # first-round pseudo-label preserved
    assert df["is_pseudo"][0] is True


def test_rollback_labels_restores_backup(tmp_path):
    """п.2.3: backup restores the pre-round labels file exactly."""
    labels = pl.DataFrame(
        {
            "action": [-100] * 5,
            "outcome": [2.0] * 5,
        }
    )
    path = tmp_path / "labels.parquet"
    save_labels_parquet(labels, path)

    _update_labels_parquet(str(path), [(0, 1, 1.0)], round_idx=2)
    assert pl.read_parquet(path)["action"][0] == 1

    _rollback_labels(str(path), 2)
    restored = pl.read_parquet(path)
    assert restored["action"].to_list() == [-100] * 5
    assert "is_pseudo" not in restored.columns  # rolled back to original


def test_rollback_labels_missing_backup_raises(tmp_path):
    """п.2.3: rollback without a backup raises FileNotFoundError."""
    path = tmp_path / "labels.parquet"
    save_labels_parquet(
        pl.DataFrame(
            {
                "action": [-100],
                "outcome": [2.0],
            }
        ),
        path,
    )
    with pytest.raises(FileNotFoundError):
        _rollback_labels(str(path), 99)
