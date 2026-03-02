# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit


# ----------------------------------------------------------------------
# Numba‑compiled core functions for each pivot method.
# Each returns a tuple of arrays (always 9 elements: TP, S1..S4, R1..R4)
# For methods with fewer levels, missing ones are filled with NaN.
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _pivot_camarilla(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> tuple[np.ndarray, ...]:
    tp = (high + low + close) / 3.0
    hl_range = high - low   # non‑zero range assumed >0
    s1 = close - 11.0 / 120.0 * hl_range
    s2 = close - 11.0 / 60.0 * hl_range
    s3 = close - 0.275 * hl_range
    s4 = close - 0.55 * hl_range
    r1 = close + 11.0 / 120.0 * hl_range
    r2 = close + 11.0 / 60.0 * hl_range
    r3 = close + 0.275 * hl_range
    r4 = close + 0.55 * hl_range
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@jit(nopython=True, cache=True)
def _pivot_classic(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> tuple[np.ndarray, ...]:
    tp = (high + low + close) / 3.0
    hl_range = high - low
    s1 = 2.0 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2.0 * hl_range
    s4 = tp - 3.0 * hl_range
    r1 = 2.0 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2.0 * hl_range
    r4 = tp + 3.0 * hl_range
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@jit(nopython=True, cache=True)
def _pivot_demark(
    open_: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> tuple[np.ndarray, ...]:
    n = len(close)
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        if open_[i] == close[i]:
            tp[i] = 0.25 * (high[i] + low[i] + 2.0 * close[i])
        elif close[i] > open_[i]:
            tp[i] = 0.25 * (2.0 * high[i] + low[i] + close[i])
        else:
            tp[i] = 0.25 * (high[i] + 2.0 * low[i] + close[i])
    s1 = 2.0 * tp - high
    r1 = 2.0 * tp - low
    # Fill unused levels with NaN
    s2 = np.full(n, np.nan, dtype=np.float64)
    s3 = s2.copy()
    s4 = s2.copy()
    r2 = s2.copy()
    r3 = s2.copy()
    r4 = s2.copy()
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@jit(nopython=True, cache=True)
def _pivot_fibonacci(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> tuple[np.ndarray, ...]:
    tp = (high + low + close) / 3.0
    hl_range = high - low
    s1 = tp - 0.382 * hl_range
    s2 = tp - 0.618 * hl_range
    s3 = tp - hl_range
    r1 = tp + 0.382 * hl_range
    r2 = tp + 0.618 * hl_range
    r3 = tp + hl_range
    s4 = np.full(len(close), np.nan, dtype=np.float64)
    r4 = s4.copy()
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@jit(nopython=True, cache=True)
def _pivot_traditional(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray
) -> tuple[np.ndarray, ...]:
    tp = (high + low + close) / 3.0
    hl_range = high - low
    s1 = 2.0 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2.0 * hl_range
    s4 = tp - 2.0 * hl_range  # Note: same as s3 in original? kept for consistency
    r1 = 2.0 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2.0 * hl_range
    r4 = tp + 2.0 * hl_range
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@jit(nopython=True, cache=True)
def _pivot_woodie(
    open_: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray
) -> tuple[np.ndarray, ...]:
    tp = (2.0 * open_ + high + low) / 4.0
    hl_range = high - low
    s1 = 2.0 * tp - high
    s2 = tp - hl_range
    s3 = low - 2.0 * (high - tp)
    s4 = s3 - hl_range
    r1 = 2.0 * tp - low
    r2 = tp + hl_range
    r3 = high + 2.0 * (tp - low)
    r4 = r3 + hl_range
    return tp, s1, s2, s3, s4, r1, r2, r3, r4


# ----------------------------------------------------------------------
# Dispatch dictionary mapping method names to functions
# ----------------------------------------------------------------------
_PIVOT_FUNCTIONS = {
    "camarilla": _pivot_camarilla,
    "classic": _pivot_classic,
    "demark": _pivot_demark,
    "fibonacci": _pivot_fibonacci,
    "traditional": _pivot_traditional,
    "woodie": _pivot_woodie,
}


# ----------------------------------------------------------------------
# Helper to convert pandas frequency string to Polars interval
# ----------------------------------------------------------------------
def _anchor_to_polars_interval(anchor: str) -> str:
    """Convert anchor string (e.g., 'D', 'W', 'M') to Polars interval."""
    anchor = anchor.upper()
    mapping = {
        'D': '1d',
        'W': '1w',
        'M': '1mo',
        'Y': '1y',
        'YE': '1y',
    }
    return mapping.get(anchor, '1d')  # default to daily


# ----------------------------------------------------------------------
# Main public function
# ----------------------------------------------------------------------
def pivots_ind(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    method: str = "traditional",
    anchor: str = "D",
) -> pl.DataFrame:
    """
    Calculate Pivot Points (support/resistance levels) for the given price data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing price columns and a datetime column.
    open_col, high_col, low_col, close_col : str
        Names of the columns with OHLC prices.
    date_col : str
        Name of the datetime column (must be sorted).
    method : str
        Pivot calculation method. One of:
        'traditional', 'fibonacci', 'woodie', 'classic', 'demark', 'camarilla'.
    anchor : str
        Resampling frequency (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
        Must be a valid pandas offset string (since we rely on Polars' dynamic grouping,
        we convert it to a Polars interval; only basic frequencies are supported).

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the same rows as the input, plus columns:
        P (pivot), S1..S4 (support levels), R1..R4 (resistance levels).
        The columns are named e.g. 'PIVOTS_TRAD_D_P', 'PIVOTS_TRAD_D_S1', etc.
    """
    # Validate method
    method = method.lower()
    if method not in _PIVOT_FUNCTIONS:
        raise ValueError(f"Unknown pivot method: {method}. \
            Choose from {list(_PIVOT_FUNCTIONS.keys())}")
    # Ensure datetime is sorted
    df = df.sort(date_col)
    # Convert anchor to Polars interval
    interval = _anchor_to_polars_interval(anchor)
    # Step 1: resample to anchor frequency using group_by_dynamic
    # We need to aggregate: first open, max high, min low, last close.
    resampled = (
        df.group_by_dynamic(
            index_column=date_col,
            every=interval,
            closed="left",
            include_boundaries=False,
        )
        .agg([
            pl.col(open_col).first().alias(f"{open_col}_agg"),
            pl.col(high_col).max().alias(f"{high_col}_agg"),
            pl.col(low_col).min().alias(f"{low_col}_agg"),
            pl.col(close_col).last().alias(f"{close_col}_agg"),
        ])
    )
    # Extract aggregated arrays for Numba calculation
    np_open = resampled[f"{open_col}_agg"].to_numpy().astype(np.float64)
    np_high = resampled[f"{high_col}_agg"].to_numpy().astype(np.float64)
    np_low = resampled[f"{low_col}_agg"].to_numpy().astype(np.float64)
    np_close = resampled[f"{close_col}_agg"].to_numpy().astype(np.float64)
    # Call the appropriate pivot function
    pivot_func = _PIVOT_FUNCTIONS[method]
    if method in ("demark", "woodie"):
        # These functions require open_
        results = pivot_func(np_open, np_high, np_low, np_close)
    else:
        # Other functions require only high, low, close
        results = pivot_func(np_high, np_low, np_close)
    # Unpack results (always 9 arrays)
    tp_arr, s1_arr, s2_arr, s3_arr, s4_arr, r1_arr, r2_arr, r3_arr, r4_arr = results
    # Add computed columns to resampled DataFrame
    suffix = f"_{method[:4].upper()}_{anchor}"
    resampled = resampled.with_columns([
        pl.Series(f"PIVOTS{suffix}_P", tp_arr),
        pl.Series(f"PIVOTS{suffix}_S1", s1_arr),
        pl.Series(f"PIVOTS{suffix}_S2", s2_arr),
        pl.Series(f"PIVOTS{suffix}_S3", s3_arr),
        pl.Series(f"PIVOTS{suffix}_S4", s4_arr),
        pl.Series(f"PIVOTS{suffix}_R1", r1_arr),
        pl.Series(f"PIVOTS{suffix}_R2", r2_arr),
        pl.Series(f"PIVOTS{suffix}_R3", r3_arr),
        pl.Series(f"PIVOTS{suffix}_R4", r4_arr),
    ])
    # Step 2: shift the index forward by one period (as in original)
    # In Polars we can do this by adding the interval to the date column.
    shifted = resampled.with_columns(
        (pl.col(date_col).dt.offset_by(interval)).alias(f"{date_col}_shifted")
    )
    # Step 3: forward‑fill the pivot values back to the original dates
    # We need to join the shifted pivot table with the original df on date,
    # carrying forward the nearest pivot values.
    # This is a classic asof join: for each original date, find the most recent
    # pivot date that is ≤ original date.
    result = df.join_asof(
        shifted.select([
            pl.col(f"{date_col}_shifted").alias(date_col),
            pl.col(f"PIVOTS{suffix}_P"),
            pl.col(f"PIVOTS{suffix}_S1"),
            pl.col(f"PIVOTS{suffix}_S2"),
            pl.col(f"PIVOTS{suffix}_S3"),
            pl.col(f"PIVOTS{suffix}_S4"),
            pl.col(f"PIVOTS{suffix}_R1"),
            pl.col(f"PIVOTS{suffix}_R2"),
            pl.col(f"PIVOTS{suffix}_R3"),
            pl.col(f"PIVOTS{suffix}_R4"),
        ]),
        on=date_col,
        strategy="forward",
    )
    # Optionally drop rows where pivot values are all NaN 
    # (original behaviour for some methods)
    # But we keep them as is.
    return result