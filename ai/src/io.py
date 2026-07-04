from __future__ import annotations

import polars as pl

from .datatypes import OrderBlock


def load_features_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_labels_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def save_labels_parquet(df_labels: pl.DataFrame, path: str) -> None:
    df_labels.write_parquet(path)


def load_order_blocks_parquet(path: str) -> list[OrderBlock]:
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
