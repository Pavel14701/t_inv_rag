# -*- coding: utf-8 -*-
"""Ichimoku Kinko Hyo (Ichimoku Cloud) indicator.

Provides a Numba-accelerated core and a Polars integration.

The indicator consists of five components:
    - Tenkan-sen (Conversion Line): midprice over `tenkan` periods.
    - Kijun-sen (Base Line): midprice over `kijun` periods.
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2,
      shifted forward by `kijun` periods.
    - Senkou Span B (Leading Span B): midprice over `senkou` periods,
      shifted forward by `kijun` periods.
    - Chikou Span (Lagging Span): close shifted backward
      by `kijun` periods.

All floating-point operations follow IEEE 754 rules. Infinite values are
replaced with NaN before calculation, and NaN inside any window propagates
to the result.
"""

from datetime import timedelta

import numpy as np
import polars as pl

from numba import njit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)


@njit(cache=True)
def _midprice_multi_numba(
    high: np.ndarray,
    low: np.ndarray,
    len1: int,
    len2: int,
    len3: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three midprices for three window lengths in a single pass.

    This function avoids scanning the input arrays three separate times,
    significantly improving performance for large datasets. The algorithm
    performs a simple O(n * max_len) loop, which is still very fast
    because the windows are typically small (<= 52) and the code is
    compiled with Numba.

    Parameters
    ----------
    high : np.ndarray
        High prices (float64).
    low : np.ndarray
        Low prices (float64).
    len1, len2, len3 : int
        Window lengths for the three midprices (e.g. Tenkan, Kijun, Senkou).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Three arrays of the same length as `high`, containing the
        midprices for each respective window. The first `(len_i - 1)`
        values of each array are NaN.

    Notes
    -----
    - If any NaN appears in a window (in `high` or `low`), the result for
      that window is NaN (IEEE 754 propagation).
    - If `n < max(len1, len2, len3)`, all returned arrays are NaN.

    """
    n = len(high)
    out1 = np.full(n, np.nan, dtype=np.float64)
    out2 = np.full(n, np.nan, dtype=np.float64)
    out3 = np.full(n, np.nan, dtype=np.float64)
    max_len = max(len1, len2, len3)
    if n < max_len:
        return out1, out2, out3
    # Iterate over all positions; the per-window guards below skip
    # positions where the window is not yet complete (warmup).
    for i in range(n):
        # Window 1
        if i - len1 + 1 >= 0:
            max_h1 = -np.inf
            min_l1 = np.inf
            valid1 = True
            for j in range(i - len1 + 1, i + 1):
                if np.isnan(high[j]) or np.isnan(low[j]):
                    valid1 = False
                    break
                if high[j] > max_h1:
                    max_h1 = high[j]
                if low[j] < min_l1:
                    min_l1 = low[j]
            if valid1:
                out1[i] = (max_h1 + min_l1) * 0.5
        # Window 2
        if i - len2 + 1 >= 0:
            max_h2 = -np.inf
            min_l2 = np.inf
            valid2 = True
            for j in range(i - len2 + 1, i + 1):
                if np.isnan(high[j]) or np.isnan(low[j]):
                    valid2 = False
                    break
                if high[j] > max_h2:
                    max_h2 = high[j]
                if low[j] < min_l2:
                    min_l2 = low[j]
            if valid2:
                out2[i] = (max_h2 + min_l2) * 0.5
        # Window 3
        if i - len3 + 1 >= 0:
            max_h3 = -np.inf
            min_l3 = np.inf
            valid3 = True
            for j in range(i - len3 + 1, i + 1):
                if np.isnan(high[j]) or np.isnan(low[j]):
                    valid3 = False
                    break
                if high[j] > max_h3:
                    max_h3 = high[j]
                if low[j] < min_l3:
                    min_l3 = low[j]
            if valid3:
                out3[i] = (max_h3 + min_l3) * 0.5
    return out1, out2, out3


@njit(cache=True)
def _shift_forward(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D array forward by a given number of positions,
    filling the beginning with NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input float64 array.
    shift : int
        Number of positions to shift forward (must be non-negative).

    Returns
    -------
    np.ndarray
        New array of the same length as `arr`. The first `shift` elements
        are NaN, and the remaining elements come from `arr` truncated
        at the end.

    """
    if shift <= 0:
        return arr.copy()  # return a copy to avoid aliasing
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if shift < n:
        out[shift:] = arr[:-shift]
    return out


def ichimoku_core_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    include_chikou: bool = True,
    lookahead: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Core Ichimoku calculation returning raw numpy arrays.

    This function computes all five Ichimoku components:
        - Tenkan-sen (Conversion Line)
        - Kijun-sen (Base Line)
        - Senkou Span A (Leading Span A)
        - Senkou Span B (Leading Span B)
        - Chikou Span (Lagging Span) - optional, depending on flags

    All midprice calculations are performed in a single pass for efficiency.
    The returned arrays have not yet been shifted forward for the Senkou spans,
    and no global offset or fillna has been applied. Use the helper functions
    below to post-process the results.

    Parameters
    ----------
    high, low, close : np.ndarray
        1D float64 arrays of price data. Must have the same length.
    tenkan, kijun, senkou : int
        Window lengths for the three midprice calculations (must be >= 1).
    include_chikou : bool
        If True, the Chikou Span is computed
        (requires lookahead=True to avoid future data).
    lookahead : bool
        If False, Chikou Span is omitted even if `include_chikou` is True,
        preventing any lookahead bias.

    Returns
    -------
    tuple containing:
        tenkan_sen : np.ndarray
            Conversion line (midprice over `tenkan` periods).
        kijun_sen : np.ndarray
            Base line (midprice over `kijun` periods).
        span_a : np.ndarray
            Senkou Span A = (tenkan_sen + kijun_sen) / 2.
        span_b : np.ndarray
            Senkou Span B = midprice over `senkou` periods.
        chikou_span : np.ndarray or None
            Chikou Span = `close` shifted backward by `kijun` periods,
            i.e. chikou[i] = close[i + kijun], or None if not requested.

    Raises
    ------
    ValueError
        If any period is < 1 or the input arrays have different lengths.

    Notes
    -----
    - If `len(high) < max(tenkan, kijun, senkou)`, all midprice arrays
      are NaN (graceful handling).
    - NaN values in the input propagate to the affected windows.

    """
    if tenkan < 1 or kijun < 1 or senkou < 1:
        raise ValueError('tenkan, kijun and senkou must all be >= 1')
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            'high, low and close must have the same length: '
            f'got {len(high)}, {len(low)}, {len(close)}.'
        )

    tenkan_sen, kijun_sen, span_b = _midprice_multi_numba(
        high, low, tenkan, kijun, senkou
    )
    span_a = (tenkan_sen + kijun_sen) * 0.5
    if include_chikou and lookahead:
        shift = kijun
        n = len(close)
        chikou = np.full(n, np.nan, dtype=np.float64)
        if shift < n:
            chikou[:-shift] = close[shift:]
        chikou_span = chikou
    else:
        chikou_span = None
    return tenkan_sen, kijun_sen, span_a, span_b, chikou_span


def ichimoku_ind(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    date_col: str | None = 'date',
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    include_chikou: bool = True,
    lookahead: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute Ichimoku Cloud indicator and return two Polars DataFrames.

    This function is the main entry point for Polars users. It extracts the required
    columns, runs the Numba-optimised core, applies the necessary forward shifts
    for Senkou Spans, adds a final offset (if requested), fills missing values,
    and constructs two DataFrames:

    * **Historical DataFrame** - contains all five Ichimoku
    lines aligned with the original index.
    * **Forward-looking DataFrame** - contains the future Senkou Spans (
        projected forward
      by `kijun` periods). This DataFrame uses either dates (if `date_col` is provided
      and temporal) or an integer index starting after the last row of the input.

    Parameters
    ----------
    df : pl.DataFrame
        Input data with at least columns for high, low, close, and optionally date.
    high_col, low_col, close_col : str
        Names of the columns containing high, low and close prices.
    date_col : str or None
        Name of the column with dates (used to create the forward index).
        If None, or if the column is not temporal (Date/Datetime),
        an integer index is used.
    tenkan, kijun, senkou : int
        Periods for the three midprice calculations (must be >= 1).
    include_chikou : bool
        Whether to compute the Chikou Span in the historical DataFrame.
    lookahead : bool
        If False, the Chikou Span is omitted to prevent any lookahead bias.
    offset : int
        Global shift applied to all components *after* the standard Ichimoku shifts.
        Positive values shift forward (future), negative shift backward.
    fillna : float, optional
        Value to replace any remaining NaNs after shifting.
        If None, NaNs are left as is.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input columns:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        - **historical** : Polars DataFrame with columns:
            `ITS_{tenkan}` (Tenkan-sen),
            `IKS_{kijun}` (Kijun-sen),
            `ISA_{tenkan}` (Senkou Span A, already shifted forward),
            `ISB_{senkou}` (Senkou Span B, already shifted forward),
            and optionally `ICS_{kijun}` (Chikou Span) if `include_chikou` is True.
        - **forward** : Polars DataFrame with columns for the future Senkou Spans:
            `ISA_{tenkan}` and `ISB_{senkou}`, indexed either by dates or integers.

    Raises
    ------
    ValueError
        If any period is < 1, the price columns have different lengths,
        the input contains NaN with `nan_policy='raise'`,
        or the series is too short for the requested periods.

    Notes
    -----
    - Infinites are replaced with NaN before any calculation.
    - This function is IEEE 754 compliant.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 120
    >>> close = 100 + np.cumsum(rng.standard_normal(n))
    >>> df = pl.DataFrame({
    ...     "date": pl.date_range(date(2020, 1, 1), date(2020, 4, 29), "1d", eager=True),
    ...     "high": close + np.abs(rng.standard_normal(n)),
    ...     "low": close - np.abs(rng.standard_normal(n)),
    ...     "close": close,
    ... })
    >>> hist, fwd = ichimoku_ind(df, tenkan=9, kijun=26, senkou=52)
    >>> print(hist)
    >>> print(fwd)

    """  # noqa: E501
    if tenkan < 1 or kijun < 1 or senkou < 1:
        raise ValueError('tenkan, kijun and senkou must all be >= 1')

    # 1. Extract numpy arrays with minimal copying
    high = df[high_col].to_numpy().astype(np.float64, copy=True)
    low = df[low_col].to_numpy().astype(np.float64, copy=True)
    close = df[close_col].to_numpy().astype(np.float64, copy=True)
    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            'high, low and close columns must have the same length: '
            f'got {len(high)}, {len(low)}, {len(close)}.'
        )
    n = len(close)
    min_len = max(tenkan, kijun, senkou)
    if n < min_len:
        raise ValueError(
            f'Input series too short: need at least {min_len} elements, '
            f'got {n}.'
        )

    # Replace infinities with NaN, then apply the NaN policy
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)
    high = _handle_nan_policy(high, nan_policy, high_col)
    low = _handle_nan_policy(low, nan_policy, low_col)
    close = _handle_nan_policy(close, nan_policy, close_col)

    # Ensure C-contiguous (required by some Numba operations)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    # 2. Compute core arrays (no shifts yet)
    tenkan_sen, kijun_sen, span_a, span_b, chikou = ichimoku_core_numba(
        high, low, close,
        tenkan=tenkan,
        kijun=kijun,
        senkou=senkou,
        include_chikou=include_chikou,
        lookahead=lookahead,
    )
    # 3. Apply Senkou forward shift (kijun) - these become the cloud boundaries
    span_a_shifted = _shift_forward(span_a, kijun)
    span_b_shifted = _shift_forward(span_b, kijun)
    # 4. Apply global offset and fillna to every component
    tenkan_final = _apply_offset_fillna(tenkan_sen, offset, fillna)
    kijun_final = _apply_offset_fillna(kijun_sen, offset, fillna)
    span_a_final = _apply_offset_fillna(span_a_shifted, offset, fillna)
    span_b_final = _apply_offset_fillna(span_b_shifted, offset, fillna)
    if chikou is not None:
        chikou_final = _apply_offset_fillna(chikou, offset, fillna)
    else:
        chikou_final = None

    # 5. Build the historical DataFrame
    hist_columns = {
        f'ITS_{tenkan}': tenkan_final,
        f'IKS_{kijun}': kijun_final,
        f'ISA_{tenkan}': span_a_final,
        f'ISB_{senkou}': span_b_final,
    }
    if chikou_final is not None:
        hist_columns[f'ICS_{kijun}'] = chikou_final
    hist_df = pl.DataFrame(hist_columns)

    # 6. Build the forward-looking DataFrame (future Senkou Spans)
    #    The last `kijun` values of the *unshifted* span_a and span_b
    #    are exactly the values that will appear in the future.
    last_span_a = span_a[-kijun:].copy()
    last_span_b = span_b[-kijun:].copy()
    if fillna is not None:
        replace_inf_with_nan(last_span_a)
        replace_inf_with_nan(last_span_b)
        last_span_a = _apply_offset_fillna(last_span_a, 0, fillna)
        last_span_b = _apply_offset_fillna(last_span_b, 0, fillna)

    date_is_temporal = (
        date_col is not None
        and date_col in df.columns
        and df[date_col].dtype in (pl.Date, pl.Datetime)
    )
    if date_is_temporal and date_col is not None:
        date_name = date_col
        last_date = df[date_name].item(-1)
        # Create a range of future dates starting from the next day
        future_dates = pl.date_range(
            start=last_date + timedelta(days=1),
            end=last_date + timedelta(days=kijun),
            interval='1d',
            eager=True,
        )
        forward_df = pl.DataFrame({
            'date': future_dates,
            f'ISA_{tenkan}': last_span_a,
            f'ISB_{senkou}': last_span_b,
        })
    else:
        # Use integer index starting from the current length
        start_idx = len(df)
        forward_df = pl.DataFrame({
            f'ISA_{tenkan}': last_span_a,
            f'ISB_{senkou}': last_span_b,
        }).with_row_index('index', offset=start_idx)
    return hist_df, forward_df