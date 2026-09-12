# -*- coding: utf-8 -*-
"""Ready-made per-timeframe configurations for order block detection.

Tuning rationale
----------------
- Low timeframes (1m/5m): noisy, mean-reverting -> tight ZigZag
  (small distance/prominence), short dynamic-lookback bounds, short
  confirmation windows, clustering on to merge overlapping zones.
- Mid timeframes (15m/1h): balanced defaults with ADX trend filter.
- High timeframes (4h/1d): strong trends -> wider ZigZag, ADX + market
  structure filters, RSI confirmation, age penalty for freshness.
All presets use the repaint-free online ZigZag.
"""
from __future__ import annotations

from .config import OrderBlockConfig


#: Canonical timeframe -> preset mapping keys (aliases are resolved by
#: :func:`get_order_block_config`).
TIMEFRAME_CONFIGS: dict[str, OrderBlockConfig] = {
    # --- scalping: noise dominates, keep it fast and tight -------------
    '1m': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.0015,
        zigzag_distance=3,
        min_extreme_gap=3,
        lookback_min=3,
        lookback_max=20,
        confirmation_window=8,
        volume_window=30,
        cluster_blocks=True,
        max_cluster_time_gap=None,
    ),
    '5m': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.003,
        zigzag_distance=4,
        min_extreme_gap=4,
        lookback_min=5,
        lookback_max=30,
        confirmation_window=10,
        volume_window=24,
        cluster_blocks=True,
    ),
    # --- intraday: balanced defaults + trend filter --------------------
    '15m': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.005,
        zigzag_distance=5,
        min_extreme_gap=5,
        use_adx_filter=True,
        adx_period=14,
        adx_threshold=20.0,
    ),
    '1h': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.008,
        zigzag_distance=6,
        min_extreme_gap=6,
        use_adx_filter=True,
        use_rsi_confirmation=True,
        strength_age_penalty=True,
    ),
    # --- swing: trends dominate, favour freshness and structure --------
    '4h': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.012,
        zigzag_distance=8,
        min_extreme_gap=8,
        use_adx_filter=True,
        adx_threshold=25.0,
        use_market_structure_filter=True,
        structure_lookback=8,
        use_rsi_confirmation=True,
        strength_age_penalty=True,
        require_complete_window=True,
    ),
    '1d': OrderBlockConfig(
        use_online_extremes=True,
        online_reversal_pct=0.02,
        zigzag_distance=10,
        min_extreme_gap=10,
        use_adx_filter=True,
        adx_threshold=25.0,
        use_market_structure_filter=True,
        use_rsi_confirmation=True,
        max_extreme_age=30,
        strength_age_penalty=True,
        strength_age_halflife=40,
        require_complete_window=True,
    ),
}

# Accepted aliases for each canonical key
_ALIASES: dict[str, str] = {}
for _key in TIMEFRAME_CONFIGS:
    _ALIASES[_key] = _key
    _ALIASES[_key.upper()] = _key
    _ALIASES[_key + 's'] = _key
for _alt, _key in {
    'm1': '1m', 'm5': '5m', 'm15': '15m', 'h1': '1h', 'h4': '4h',
    'd1': '1d', '1min': '1m', '5min': '5m', '15min': '15m',
    '60min': '1h', '60m': '1h', '240m': '4h', '1day': '1d', 'D': '1d',
    '1D': '1d', '4H': '4h', '1H': '1h', '15M': '15m', '5M': '5m',
    '1M': '1m', 'M1': '1m', 'M5': '5m', 'M15': '15m', 'H1': '1h',
    'H4': '4h',
}.items():
    _ALIASES[_alt] = _key


def get_order_block_config(timeframe: str) -> OrderBlockConfig:
    """Return a fresh preset config for ``timeframe``.

    Raises
    ------
    ValueError
        If the timeframe is not recognised.

    """
    key = _ALIASES.get(timeframe)
    if key is None:
        supported = ', '.join(sorted(TIMEFRAME_CONFIGS))
        raise ValueError(
            f'Unsupported timeframe {timeframe!r}; '
            f'supported: {supported}'
        )
    return OrderBlockConfig(**vars(TIMEFRAME_CONFIGS[key]))


__all__ = [
    'TIMEFRAME_CONFIGS',
    'get_order_block_config',
]
