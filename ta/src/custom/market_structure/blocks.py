# -*- coding: utf-8 -*-
"""Main entry point: :func:`identify_order_blocks`."""
from __future__ import annotations

import numpy as np
import polars as pl

from ...trend.zigzag import zigzag_peaks_valleys
from .candidates import generate_block_candidates
from .clustering import cluster_order_blocks
from .config import OrderBlockConfig
from .indicators import precompute_indicators
from .online_zigzag import OnlineZigZag, confirmed_pivot_arrays
from .types import OrderBlock
from .validation import validate_block_candidates


def identify_order_blocks(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
    date_col: str = 'date',
    cfg: OrderBlockConfig | None = None,
) -> pl.DataFrame:
    """Detect confirmed order blocks on OHLCV data.

    With ``cfg.use_online_extremes`` the pipeline uses the incremental
    ZigZag: only pivots that were already final before the breakout bar
    can produce a block, which removes ZigZag repainting (look-ahead
    bias) from the results.
    """
    if cfg is None:
        cfg = OrderBlockConfig()
    if cfg.max_history is not None:
        df = df.tail(cfg.max_history)
    high = df[high_col].to_numpy().astype(np.float64)
    low = df[low_col].to_numpy().astype(np.float64)
    close = df[close_col].to_numpy().astype(np.float64)
    volume = df[volume_col].to_numpy().astype(np.float64)
    # python datetimes (not np.datetime64) so blocks can be assembled
    # into a Polars frame without object-cast issues
    dates = df[date_col].to_list()

    pivot_confirm: dict[int, int] | None = None
    pivot_next_extreme: dict[int, int] | None = None
    if cfg.use_online_extremes:
        zz = OnlineZigZag(
            cfg.effective_online_reversal, cfg.online_reversal_pct,
        )
        pivots = zz.update_series(high, low)
        peak_indices, valley_indices, confirm, nxt = confirmed_pivot_arrays(
            pivots,
        )
        pivot_confirm = {p.idx: p.confirm_idx for p in pivots}
        pivot_next_extreme = {
            p.idx: int(n) for p, n in zip(pivots, nxt)
        }
    else:
        peak_indices, valley_indices = zigzag_peaks_valleys(
            high, low,
            prominence_peak=cfg.zigzag_prominence_peak,
            prominence_valley=cfg.zigzag_prominence_valley,
            distance=cfg.zigzag_distance,
            width=cfg.zigzag_width,
            wlen=cfg.zigzag_wlen,
            rel_height=cfg.zigzag_rel_height,
            plateau_size=cfg.zigzag_plateau_size,
        )
    indicators = precompute_indicators(high, low, close, volume, cfg)
    candidates = generate_block_candidates(
        high, low, close, volume, dates,
        peak_indices, valley_indices, indicators, cfg,
    )
    confirmed = validate_block_candidates(
        high, low, close, volume, dates,
        candidates, indicators, cfg,
        peak_indices, valley_indices, [],
        pivot_confirm=pivot_confirm,
        pivot_next_extreme=pivot_next_extreme,
    )
    if cfg.cluster_blocks and confirmed:
        confirmed = cluster_order_blocks(
            confirmed,
            cfg.cluster_price_tolerance,
            cfg.max_cluster_time_gap,
        )
    confirmed.sort(key=lambda b: b.start)
    if not confirmed:
        return _empty_block_frame()
    data = {
        'id': [b.id for b in confirmed],
        'block_type': [b.block_type for b in confirmed],
        'start': [b.start for b in confirmed],
        'break': [b.break_ for b in confirmed],
        'retest': [b.retest for b in confirmed],
        'zone_low': [b.zone_low for b in confirmed],
        'zone_high': [b.zone_high for b in confirmed],
        'strength': [b.strength for b in confirmed],
        'structure_label': [
            b.structure_label for b in confirmed
        ],
        'trend_direction': [
            b.trend_direction for b in confirmed
        ],
    }
    return pl.DataFrame(data).sort('start')


def _empty_block_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            'id': pl.Int64,
            'block_type': pl.Utf8,
            'start': pl.Datetime,
            'break': pl.Datetime,
            'retest': pl.Datetime,
            'zone_low': pl.Float64,
            'zone_high': pl.Float64,
            'strength': pl.Float64,
            'structure_label': pl.Utf8,
            'trend_direction': pl.Utf8,
        }
    )


__all__ = [
    'identify_order_blocks',
    'OrderBlock',
    'OrderBlockConfig',
]
