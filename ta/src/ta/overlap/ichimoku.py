# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _rolling_max_numba(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Compute the rolling maximum over a fixed window.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D float64 array.
    length : int
        Window size.

    Returns
    -------
    np.ndarray
        Array of same length as `arr`. The first `length-1` positions are NaN,
        the rest contain the maximum of the last `length` elements.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        out[i] = np.max(arr[i - length + 1:i + 1])
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _rolling_min_numba(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Compute the rolling minimum over a fixed window.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D float64 array.
    length : int
        Window size.

    Returns
    -------
    np.ndarray
        Array of same length as `arr`. The first `length-1` positions are NaN,
        the rest contain the minimum of the last `length` elements.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        out[i] = np.min(arr[i - length + 1:i + 1])
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _midprice_multi_numba(
    high: np.ndarray,
    low: np.ndarray,
    len1: int,
    len2: int,
    len3: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute three midprices for three different window lengths in a single pass.

    This function avoids scanning the input arrays three separate times,
    significantly improving performance for large datasets. The algorithm
    performs a simple O(n * max_len) loop, which is still very fast because
    the windows are typically small (≤ 52) and the code is compiled with Numba.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Three arrays of the same length as `high`, containing the midprices for
        each respective window. The first `(len_i - 1)` values of each array are NaN.
    """
    n = len(high)
    max_len = max(len1, len2, len3)
    out1 = np.full(n, np.nan, dtype=np.float64)
    out2 = np.full(n, np.nan, dtype=np.float64)
    out3 = np.full(n, np.nan, dtype=np.float64)
    if n < max_len:
        return out1, out2, out3
    for i in range(max_len - 1, n):
        # Window 1
        if i - len1 + 1 >= 0:
            max_h1 = high[i - len1 + 1]
            min_l1 = low[i - len1 + 1]
            for j in range(i - len1 + 2, i + 1):
                if high[j] > max_h1:
                    max_h1 = high[j]
                if low[j] < min_l1:
                    min_l1 = low[j]
            out1[i] = (max_h1 + min_l1) * 0.5
        # Window 2
        if i - len2 + 1 >= 0:
            max_h2 = high[i - len2 + 1]
            min_l2 = low[i - len2 + 1]
            for j in range(i - len2 + 2, i + 1):
                if high[j] > max_h2:
                    max_h2 = high[j]
                if low[j] < min_l2:
                    min_l2 = low[j]
            out2[i] = (max_h2 + min_l2) * 0.5
        # Window 3
        if i - len3 + 1 >= 0:
            max_h3 = high[i - len3 + 1]
            min_l3 = low[i - len3 + 1]
            for j in range(i - len3 + 2, i + 1):
                if high[j] > max_h3:
                    max_h3 = high[j]
                if low[j] < min_l3:
                    min_l3 = low[j]
            out3[i] = (max_h3 + min_l3) * 0.5
    return out1, out2, out3


@jit(nopython=True, fastmath=True, cache=True)
def _shift_forward(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift a 1D array forward by a given number of positions,
    filling the beginning with NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input float64 array.
    shift : int
        Number of positions to shift forward (must be non‑negative).

    Returns
    -------
    np.ndarray
        New array of the same length as `arr`. The first `shift` elements are NaN,
        and the remaining elements come from `arr` truncated at the end.
    """
    if shift <= 0:
        return arr.copy()  # return a copy to avoid aliasing
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
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
    lookahead: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Core Ichimoku calculation returning raw numpy arrays (no offset, no fillna).

    This function computes all five Ichimoku components:
        - Tenkan‑sen (Conversion Line)
        - Kijun‑sen (Base Line)
        - Senkou Span A (Leading Span A)
        - Senkou Span B (Leading Span B)
        - Chikou Span (Lagging Span) – optional, depending on flags

    All midprice calculations are performed in a single pass for maximum efficiency.
    The returned arrays have not yet been shifted forward for the Senkou spans,
    and no global offset or fillna has been applied. Use the helper functions
    below to post‑process the results.

    Parameters
    ----------
    high, low, close : np.ndarray
        1D float64 arrays of price data. Must have the same length.
    tenkan, kijun, senkou : int
        Window lengths for the three midprice calculations.
    include_chikou : bool
        If True, the Chikou Span is computed
        (requires lookahead=True to avoid future data).
    lookahead : bool
        If False, Chikou Span is omitted even if `include_chikou` is True,
        preventing any lookahead bias.

    Returns
    -------
    Tuple containing:
        tenkan_sen : np.ndarray
            Conversion line (midprice over `tenkan` periods).
        kijun_sen : np.ndarray
            Base line (midprice over `kijun` periods).
        span_a : np.ndarray
            Senkou Span A = (tenkan_sen + kijun_sen) / 2.
        span_b : np.ndarray
            Senkou Span B = midprice over `senkou` periods.
        chikou_span : np.ndarray or None
            Chikou Span = `close` shifted backward by `(kijun - 1)` periods,
            or None if not requested.
    """
    # Ensure we have enough data; minimal length is max(tenkan, kijun, senkou)
    # The midprice functions already handle short arrays gracefully (return all NaN).
    tenkan_sen, kijun_sen, span_b = _midprice_multi_numba(
        high, low, tenkan, kijun, senkou
    )
    span_a = (tenkan_sen + kijun_sen) * 0.5
    if include_chikou and lookahead:
        shift = kijun - 1
        n = len(close)
        chikou = np.full(n, np.nan, dtype=np.float64)
        if shift > 0:
            chikou[:-shift] = close[shift:]
        chikou_span = chikou
    else:
        chikou_span = None
    return tenkan_sen, kijun_sen, span_a, span_b, chikou_span


def ichimoku_ind(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str | None = "date",
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    include_chikou: bool = True,
    lookahead: bool = True,
    offset: int = 0,
    fillna: float | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute Ichimoku Cloud indicator and return two Polars DataFrames.

    This function is the main entry point for Polars users. It extracts the required
    columns, runs the Numba‑optimised core, applies the necessary forward shifts
    for Senkou Spans, adds a final offset (if requested), fills missing values,
    and constructs two DataFrames:

    * **Historical DataFrame** – contains all five Ichimoku 
    lines aligned with the original index.
    * **Forward‑looking DataFrame** – contains the future Senkou Spans (
        projected forward
      by `kijun` periods). This DataFrame uses either dates (if `date_col` is provided)
      or an integer index starting after the last row of the input.

    Parameters
    ----------
    df : pl.DataFrame
        Input data with at least columns for high, low, close, and optionally date.
    high_col, low_col, close_col : str
        Names of the columns containing high, low and close prices.
    date_col : str or None
        Name of the column with dates (used to create the forward index).
        If None, an integer index is used.
    tenkan, kijun, senkou : int
        Periods for the three midprice calculations.
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

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        - **historical** : Polars DataFrame with columns:
            `ITS_{tenkan}` (Tenkan‑sen),
            `IKS_{kijun}` (Kijun‑sen),
            `ISA_{tenkan}` (Senkou Span A, already shifted forward),
            `ISB_{senkou}` (Senkou Span B, already shifted forward),
            and optionally `ICS_{kijun}` (Chikou Span) if `include_chikou` is True.
        - **forward** : Polars DataFrame with columns for the future Senkou Spans:
            `ISA_{tenkan}` and `ISB_{senkou}`, indexed either by dates or integers.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> df = pl.DataFrame({
    ...     "date": pl.date_range(date(2020,1,1), date(2020,12,31), "1d", eager=True),
    ...     "high": np.random.randn(366).cumsum() + 100,
    ...     "low": np.random.randn(366).cumsum() + 99,
    ...     "close": np.random.randn(366).cumsum() + 99.5,
    ... })
    >>> hist, fwd = ichimoku(df, tenkan=9, kijun=26, senkou=52)
    >>> print(hist)
    >>> print(fwd)
    """
    # 1. Extract numpy arrays with minimal copying
    high = df[high_col].to_numpy().astype(np.float64, copy=False)
    low = df[low_col].to_numpy().astype(np.float64, copy=False)
    close = df[close_col].to_numpy().astype(np.float64, copy=False)
    # Ensure C‑contiguous (required by some Numba operations)
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
        lookahead=lookahead
    )
    # 3. Apply Senkou forward shift (kijun - 1) – these become the cloud boundaries
    span_a_shifted = _shift_forward(span_a, kijun - 1)
    span_b_shifted = _shift_forward(span_b, kijun - 1)
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
        f"ITS_{tenkan}": tenkan_final,
        f"IKS_{kijun}": kijun_final,
        f"ISA_{tenkan}": span_a_final,
        f"ISB_{senkou}": span_b_final,
    }
    if chikou_final is not None:
        hist_columns[f"ICS_{kijun}"] = chikou_final
    hist_df = pl.DataFrame(hist_columns)
    # 6. Build the forward‑looking DataFrame (future Senkou Spans)
    #    The last `kijun` values of the *unshifted* span_a and span_b
    #    are exactly the values that will appear in the future.
    last_span_a = span_a[-kijun:]
    last_span_b = span_b[-kijun:]
    if date_col is not None and date_col in df.columns:
        last_date = df[date_col][-1]
        # Create a range of future dates starting from the next day
        future_dates = pl.date_range(
            start=last_date + timedelta(days=1),
            end=last_date + timedelta(days=kijun),
            interval="1d",
            eager=True
        )
        forward_df = pl.DataFrame({
            "date": future_dates,
            f"ISA_{tenkan}": last_span_a,
            f"ISB_{senkou}": last_span_b
        })
    else:
        # Use integer index starting from the current length
        start_idx = len(df)
        forward_df = pl.DataFrame({
            f"ISA_{tenkan}": last_span_a,
            f"ISB_{senkou}": last_span_b
        }).with_row_index("index", offset=start_idx)
    return hist_df, forward_df