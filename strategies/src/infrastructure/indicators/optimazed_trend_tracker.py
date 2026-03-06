# -*- coding: utf-8 -*-
"""
Optimized Trend Tracker (OTT) – Numba‑accelerated with Polars integration.
Добавлены функции для генерации сигналов напрямую из Polars DataFrame.
"""

import numpy as np
import polars as pl


def ott_signals_polars(
    df: pl.DataFrame,
    price_col: str = "close",
    ott_col: str = "OTT",
    ma_col: str = "OTT_MA",
    suffix: str = "",
) -> pl.DataFrame:
    """
    Generate OTT trading signals and add them as columns to the Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing price, OTT and MA columns.
    price_col : str
        Column name for close prices.
    ott_col : str
        Column name for OTT values.
    ma_col : str
        Column name for MA values (usually OTT_MA).
    suffix : str, optional
        Suffix for output columns (e.g., "_14_1.4_VAR").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added signal columns.
    """
    price = df[price_col].to_numpy()
    ott = df[ott_col].to_numpy()
    ma = df[ma_col].to_numpy()
    # Previous values
    price_prev = np.roll(price, 1)
    price_prev[0] = np.nan
    ott_prev = np.roll(ott, 1)
    ott_prev[0] = np.nan
    ma_prev = np.roll(ma, 1)
    ma_prev[0] = np.nan
    # Signal conditions
    buy_price_cross = (price > ott) & (price_prev <= ott_prev)
    sell_price_cross = (price < ott) & (price_prev >= ott_prev)
    buy_support_cross = (ma > ott) & (ma_prev <= ott_prev)
    sell_support_cross = (ma < ott) & (ma_prev >= ott_prev)
    buy_color_change = ott > ott_prev
    sell_color_change = ott < ott_prev
    # Combined signal
    conditions = [
        buy_price_cross,
        buy_support_cross,
        buy_color_change,
        sell_price_cross,
        sell_support_cross,
        sell_color_change,
    ]
    choices = [
        "buy_price_cross",
        "buy_support_cross",
        "buy_color_change",
        "sell_price_cross",
        "sell_support_cross",
        "sell_color_change",
    ]
    signal = np.select(conditions, choices, default="")
    # Add columns
    suffix_str = f"{suffix}" if suffix else ""
    return df.with_columns([
        pl.Series(f"buy_price_cross{suffix_str}", buy_price_cross),
        pl.Series(f"sell_price_cross{suffix_str}", sell_price_cross),
        pl.Series(f"buy_support_cross{suffix_str}", buy_support_cross),
        pl.Series(f"sell_support_cross{suffix_str}", sell_support_cross),
        pl.Series(f"buy_color_change{suffix_str}", buy_color_change),
        pl.Series(f"sell_color_change{suffix_str}", sell_color_change),
        pl.Series(f"signal{suffix_str}", signal),
    ])


def get_last_ott_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    ott_col: str = "OTT",
    ma_col: str = "OTT_MA",
) -> str | None:
    """
    Quickly get the last trading signal without computing all columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with at least two rows.
    price_col, ott_col, ma_col : column names.

    Returns
    -------
    str or None
        Signal name ('buy_price_cross', etc.) or None.
    """
    price = df[price_col].to_numpy()
    ott = df[ott_col].to_numpy()
    ma = df[ma_col].to_numpy()
    n = len(price)
    if n < 2:
        return None
    p2, p1 = price[-2], price[-1]
    o2, o1 = ott[-2], ott[-1]
    m2, m1 = ma[-2], ma[-1]

    if (p1 > o1) and (p2 <= o2):
        return "buy_price_cross"
    if (p1 < o1) and (p2 >= o2):
        return "sell_price_cross"
    if (m1 > o1) and (m2 <= o2):
        return "buy_support_cross"
    if (m1 < o1) and (m2 >= o2):
        return "sell_support_cross"
    if o1 > o2:
        return "buy_color_change"
    if o1 < o2:
        return "sell_color_change"
    return None