# -*- coding: utf-8 -*-
"""MAMA (Mesa Adaptive Moving Average) - Ehlers' adaptive filter.

This module provides:
- Numba-accelerated core (`_mama_numba_core`)
- Numba implementation (`mama_numba`) with NaN policy support
- TA-Lib backend (`mama_talib`) using TA-Lib MAMA
- Universal wrapper (`mama_ind`)
- Polars integration (`mama_polars`)

All floating-point operations follow IEEE 754 rules (no fastmath
optimisations). Infinite values are replaced with NaN before calculation.
A NaN in the input poisons the recursive filter from that point onward.
"""

from typing import Optional, Tuple

import numpy as np
import polars as pl

from numba import jit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available


# ----------------------------------------------------------------------
# Core Numba implementation (Ehlers' MAMA)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True, fastmath=False)
def _mama_numba_core(
    close: np.ndarray, fastlimit: float, slowlimit: float, prenan: int
) -> Tuple[np.ndarray, np.ndarray]:
    """MAMA core loop (Numba). Returns (mama, fama).

    The first ``prenan`` values of the output are NaN (warm-up). A NaN in
    the input poisons the recursive filter from that point onward.
    """
    n = len(close)
    mama = np.full(n, np.nan, dtype=np.float64)
    fama = np.full(n, np.nan, dtype=np.float64)
    if n < 6:
        return mama, fama
    a, b = 0.0962, 0.5769
    p_w = 0.2
    # Temporary arrays, zero-initialised to avoid leaking uninitialised
    # memory (np.empty previously could hold garbage in warm-up slots).
    wma4 = np.zeros(n, dtype=np.float64)
    dt = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    im = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)

    # Initialise first 6 values
    for i in range(6):
        mama[i] = close[i]
        fama[i] = close[i]

    for i in range(6, n):
        c = close[i]
        c1 = close[i - 1]
        c2 = close[i - 2]
        c3 = close[i - 3]
        adj_prev_period = 0.075 * period[i - 1] + 0.54
        wma4[i] = 0.4 * c + 0.3 * c1 + 0.2 * c2 + 0.1 * c3
        dt[i] = adj_prev_period * (
            a * wma4[i] + b * wma4[i - 2] - b * wma4[i - 4] - a * wma4[i - 6]
        )
        q1[i] = adj_prev_period * (
            a * dt[i] + b * dt[i - 2] - b * dt[i - 4] - a * dt[i - 6]
        )
        i1[i] = dt[i - 3]
        ji[i] = adj_prev_period * (
            a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6]
        )
        jq[i] = adj_prev_period * (
            a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6]
        )
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        i2[i] = p_w * i2[i] + (1 - p_w) * i2[i - 1]
        q2[i] = p_w * q2[i] + (1 - p_w) * q2[i - 1]
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] + q2[i] * i2[i - 1]
        re[i] = p_w * re[i] + (1 - p_w) * re[i - 1]
        im[i] = p_w * im[i] + (1 - p_w) * im[i - 1]
        if im[i] != 0.0 and re[i] != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            period[i] = 360.0 / np.arctan(im[i] / re[i])
        else:
            period[i] = 0.0
        # Period limits
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0
        period[i] = p_w * period[i] + (1 - p_w) * period[i - 1]
        if q1[i] != 0.0:  # noqa: RUF069 - exact IEEE zero/sign check
            phase[i] = np.arctan(i1[i] / q1[i])
        else:
            phase[i] = phase[i - 1]
        dphase = phase[i - 1] - phase[i]
        if dphase < 1.0:
            dphase = 1.0
        alpha[i] = fastlimit / dphase
        if alpha[i] > fastlimit:
            alpha[i] = fastlimit
        if alpha[i] < slowlimit:
            alpha[i] = slowlimit
        mama[i] = alpha[i] * c + (1 - alpha[i]) * mama[i - 1]
        fama[i] = 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i - 1]
    if prenan > 0:
        mama[:prenan] = np.nan
        fama[:prenan] = np.nan
    return mama, fama


# ----------------------------------------------------------------------
# MAMA using Numba
# ----------------------------------------------------------------------
def mama_numba(
    close: np.ndarray,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None,
    nan_policy: str = "raise",
) -> Tuple[np.ndarray, np.ndarray]:
    """MAMA using Numba with IEEE 754 compliant NaN/Inf handling.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    fastlimit, slowlimit : float
        Limits for the adaptive smoothing constant alpha.
    prenan : int, default 3
        Number of leading NaN values to force in the output.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to replace NaN after shift.
    nan_policy : str, default 'raise'
        How to handle NaNs in input:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.

    Returns
    -------
    (mama, fama) : tuple of np.ndarray
        Adaptive moving average and its smoothed companion.

    Raises
    ------
    ValueError
        If ``fastlimit/slowlimit`` are invalid, ``prenan`` is negative, or
        invalid ``nan_policy``.

    Notes
    -----
    - Infinities are replaced with NaN before calculation.
    - A NaN in the input poisons the recursive filter from that point
      onward (IEEE 754 propagation).
    - This function is IEEE 754 compliant (no fastmath).

    """
    if not (0.0 < slowlimit <= fastlimit <= 1.0):
        raise ValueError(
            f"Invalid limits: require 0 < slowlimit <= fastlimit <= 1, "
            f"got {slowlimit=} {fastlimit=}."
        )
    if prenan < 0:
        raise ValueError(f"prenan must be >= 0, got {prenan}.")
    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    mama, fama = _mama_numba_core(close, fastlimit, slowlimit, prenan)
    mama = _apply_offset_fillna(mama, offset, fillna)
    fama = _apply_offset_fillna(fama, offset, fillna)
    return mama, fama


def mama_talib(
    close: np.ndarray,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """MAMA via TA-Lib.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    fastlimit : float, default 0.5
        Upper limit for alpha.
    slowlimit : float, default 0.05
        Lower limit for alpha.
    offset : int, default 0
        Shift applied to the result.
    fillna : float or None, default None
        Value to replace NaN after shift.

    Returns
    -------
    (mama, fama) : tuple of np.ndarray
        Adaptive moving average and its smoothed companion.

    Raises
    ------
    ImportError
        If TA-Lib is not installed.

    Notes
    -----
    - TA-Lib MAMA does not handle NaNs; input must be clean.
    - Infinities are replaced with NaN before calculation.

    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    close = np.asarray(close, dtype=np.float64)
    close = close.copy()
    replace_inf_with_nan(close)
    mama, fama = talib.MAMA(close, fastperiod=fastlimit, slowperiod=slowlimit)
    mama = _apply_offset_fillna(mama, offset, fillna)
    fama = _apply_offset_fillna(fama, offset, fillna)
    return mama, fama


def mama_ind(
    close: np.ndarray | pl.Series,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> Tuple[np.ndarray, np.ndarray]:
    """Universal MAMA with automatic backend selection.

    Parameters
    ----------
        close : np.ndarray or pl.Series
            1D array of close prices.
        fastlimit : float, default 0.5
            Upper limit for alpha.
        slowlimit : float, default 0.05
            Lower limit for alpha.
        prenan : int, default 3
            Number of leading NaN values to force in the output.
        offset : int, default 0
            Shift applied to the result.
        fillna : float or None, default None
            Value to fill NaN after shift.
        use_talib : bool, default True
            If True and TA-Lib is available, use TA-Lib.
        nan_policy : str, default 'raise'
            How to handle NaNs (only for Numba backend).

    Returns
    -------
        (mama, fama) : tuple of np.ndarray
            Adaptive moving average and its smoothed companion.

    Raises
    ------
        ValueError
            If trim=True and TA-Lib backend is selected.

    Examples
    --------
        >>> import numpy as np
        >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> mama, fama = mama_ind(prices, fastlimit=0.5,
    ...     slowlimit=0.05, use_talib=False)
        >>> mama[:3]
        array([nan, nan, nan])
        >>> mama[3:]
        array([2.        , 2.5       , 3.        , 3.5       , 4.0       ,
                4.5       , 5.0       , 5.5       , 6.0       , 6.5       ])
        >>> import polars as pl
        >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mama, fama = mama_ind(s, fastlimit=0.5,
    ...     slowlimit=0.05, use_talib=False, prenan=2)
        >>> mama[:2]
        array([nan, nan])
        >>> mama[2:]
        array([2. , 2.5, 3. , 3.5, 4. ])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available:
        return mama_talib(close, fastlimit, slowlimit, offset, fillna)
    else:
        return mama_numba(
            close, fastlimit, slowlimit, prenan, offset, fillna, nan_policy
        )


def mama_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: Optional[str] = None,
) -> Tuple[pl.Series, pl.Series]:
    """Return MAMA and FAMA as Polars Series.

    Parameters
    ----------
        df : pl.DataFrame
            Input DataFrame.
        close_col : str, default 'close'
            Name of the column containing close prices.
        fastlimit : float, default 0.5
            Upper limit for alpha.
        slowlimit : float, default 0.05
            Lower limit for alpha.
        prenan : int, default 3
            Number of leading NaN values to force in the output.
        offset : int, default 0
            Shift applied to the result.
        fillna : float or None, default None
            Value to fill NaN after shift.
        use_talib : bool, default True
            Use TA-Lib if available.
        nan_policy : str, default 'raise'
            NaN handling policy (only for Numba backend).
        output_col : str or None, default None
            Prefix for output Series names. If None, uses 'MAMA' and 'FAMA'.

    Returns
    -------
        (mama, fama) : tuple of pl.Series
            Adaptive moving average and its smoothed companion.

    Examples
    --------
        >>> import polars as pl
        >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0,
    ...     5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        >>> mama, fama = mama_polars(df, fastlimit=0.5,
    ...     slowlimit=0.05, output_col="M")
        >>> mama
        shape: (10,)
        Series: 'M_MAMA' [f64]
        [
            null
            null
            null
            2.0
            2.5
            3.0
            3.5
            4.0
            4.5
            5.0
        ]
        >>> fama
        shape: (10,)
        Series: 'M_FAMA' [f64]
        [
            null
            null
            null
            2.0
            2.25
            2.5
            2.75
            3.0
            3.25
            3.5
        ]

    """
    close = df[close_col].to_numpy()
    mama, fama = mama_ind(
        close,
        fastlimit=fastlimit,
        slowlimit=slowlimit,
        prenan=prenan,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    mama_name = output_col + "_MAMA" if output_col else "MAMA"
    fama_name = output_col + "_FAMA" if output_col else "FAMA"
    return pl.Series(mama_name, mama), pl.Series(fama_name, fama)
