"""I/O utilities for persisting features, labels, and order blocks.

Provides functions to read/write Parquet files and to merge feature
and label DataFrames.
"""

from __future__ import annotations

import polars as pl

from .datatypes import OrderBlock


def load_features_parquet(path: str) -> pl.DataFrame:
    """Load feature data from a Parquet file.

    Args:
        path: Path to the .parquet file.

    Returns:
        DataFrame with feature columns. Expected to contain at least
        price, indicator, signal, and TP/SL data.

    """
    return pl.read_parquet(path)


def load_labels_parquet(path: str) -> pl.DataFrame:
    """Load label data from a Parquet file.

    Args:
        path: Path to the .parquet file.

    Returns:
        DataFrame with label columns, typically 'action' and 'outcome'.

    """
    return pl.read_parquet(path)


def save_labels_parquet(df_labels: pl.DataFrame, path: str) -> None:
    """Save label DataFrame to a Parquet file.

    Args:
        df_labels: DataFrame containing 'action' and 'outcome' columns
            (or other label columns).
        path: Destination file path (will be overwritten if exists).

    """
    df_labels.write_parquet(path)


def load_order_blocks_parquet(path: str) -> list[OrderBlock]:
    """Deserialize a list of OrderBlock objects from a Parquet file.

    The file must contain columns matching the OrderBlock dataclass
    fields. Missing optional columns are filled with defaults.

    Args:
        path: Path to the .parquet file.

    Returns:
        A list of OrderBlock instances.

    """
    df = pl.read_parquet(path)
    obs: list[OrderBlock] = []
    obs.extend(
        OrderBlock(
            id=row['id'],
            block_type=row['block_type'],
            start=row['start'],
            break_=row['break_'],
            retest=row['retest'],
            zone_low=row['zone_low'],
            zone_high=row['zone_high'],
            strength=row.get('strength', 0.0),
            structure_label=row.get('structure_label', None),
            trend_direction=row.get('trend_direction', None),
            start_idx=row.get('start_idx', -1),
            end_idx=row.get('end_idx', -1),
        )
        for row in df.iter_rows(named=True)
    )
    return obs


def merge_features_labels(
    df_feat: pl.DataFrame,
    df_lbl: pl.DataFrame,
    default_action: int = -100,
    default_outcome: float = float('nan'),
) -> pl.DataFrame:
    """Merge feature and label DataFrames, filling missing label columns.

    If both DataFrames contain a 'bar_index' column, a left join is
    performed on that column; otherwise they are concatenated column-wise
    (hstack), assuming identical row ordering.

    Missing 'action' and 'outcome' columns after the join are filled
    with ``default_action`` and ``default_outcome`` respectively.

    Args:
        df_feat: Feature DataFrame.
        df_lbl: Label DataFrame.
        default_action: Value to fill when 'action' is missing
            (default -100, which is the ignore index in loss).
        default_outcome: Value to fill when 'outcome' is missing
            (default NaN).

    Returns:
        A DataFrame containing all feature columns plus 'action' and
        'outcome' columns.

    """
    if 'bar_index' in df_feat.columns and 'bar_index' in df_lbl.columns:
        df = df_feat.join(df_lbl, on='bar_index', how='left')
    else:
        df = df_feat.join(df_lbl, how='left', on=None)
    if 'action' not in df.columns:
        df = df.with_columns(pl.lit(default_action).alias('action'))
    else:
        df = df.with_columns(pl.col('action').fill_null(default_action))
    if 'outcome' not in df.columns:
        df = df.with_columns(pl.lit(default_outcome).alias('outcome'))
    else:
        df = df.with_columns(pl.col('outcome').fill_null(default_outcome))
    return df
