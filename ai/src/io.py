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
            id=row["id"],
            block_type=row["block_type"],
            start=row["start"],
            break_=row["break_"],
            retest=row["retest"],
            zone_low=row["zone_low"],
            zone_high=row["zone_high"],
            strength=row.get("strength", 0.0),
            structure_label=row.get("structure_label", None),
            trend_direction=row.get("trend_direction", None),
            start_idx=row.get("start_idx", -1),
            end_idx=row.get("end_idx", -1),
        )
        for row in df.iter_rows(named=True)
    )
    return obs


def merge_features_labels(
    df_feat: pl.DataFrame,
    df_lbl: pl.DataFrame,
    default_action: int = -100,
    default_outcome: float = float("nan"),
) -> pl.DataFrame:
    """Merge feature and label DataFrames, filling missing label columns.

    If both DataFrames contain a 'bar_index' column, a left join is
    performed on that column. Otherwise, the DataFrames are assumed to
    have the same row order and are concatenated horizontally (hstack).

    Missing 'action' and 'outcome' columns after the merge are filled
    with ``default_action`` and ``default_outcome`` respectively.

    Args:
        df_feat: Feature DataFrame. Must contain at least the columns
            needed for features (prices, signals, etc.).
        df_lbl: Label DataFrame. Should contain at least 'action' and
            'outcome' columns, and optionally 'bar_index' for joining.
        default_action: Value to fill when 'action' is missing
            (default -100, which is the ignore index in loss).
        default_outcome: Value to fill when 'outcome' is missing
            (default NaN).

    Returns:
        A DataFrame containing all feature columns plus 'action' and
        'outcome' columns.

    """
    # 1. Check for join key
    if "bar_index" in df_feat.columns and "bar_index" in df_lbl.columns:
        df = df_feat.join(df_lbl, on="bar_index", how="left")
    else:
        # 2. No common key -> assume same row order and hstack
        if len(df_feat) != len(df_lbl):
            raise ValueError(
                f"Row count mismatch: df_feat has {len(df_feat)} rows, "
                f"df_lbl has {len(df_lbl)} rows. Cannot hstack without",
                "'bar_index'."
            )
        # Ensure action and outcome columns exist; if not, add with defaults
        lbl_cols = []
        if "action" in df_lbl.columns:
            lbl_cols.append(pl.col("action"))
        else:
            lbl_cols.append(pl.lit(default_action).alias("action"))
        if "outcome" in df_lbl.columns:
            lbl_cols.append(pl.col("outcome"))
        else:
            lbl_cols.append(pl.lit(default_outcome).alias("outcome"))
        df = df_feat.with_columns(lbl_cols)
    # 3. Ensure final columns exist (in case join did not produce them)
    if "action" not in df.columns:
        df = df.with_columns(pl.lit(default_action).alias("action"))
    else:
        df = df.with_columns(pl.col("action").fill_null(default_action))
    if "outcome" not in df.columns:
        df = df.with_columns(pl.lit(default_outcome).alias("outcome"))
    else:
        df = df.with_columns(pl.col("outcome").fill_null(default_outcome))
    return df
