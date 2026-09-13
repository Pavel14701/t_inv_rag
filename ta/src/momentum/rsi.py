# -*- coding: utf-8 -*-
"""Relative Strength Index (RSI).

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- ``+/-Inf`` on input is converted to ``NaN`` (never raises, never mutates
  the caller's array);
- ``NaN`` propagates: any price difference touching a ``NaN`` close yields
  a ``NaN`` RSI at the corresponding bars;
- ``0/0`` (flat market) and ``x/0`` are evaluated inside ``np.errstate``
  and resolved by explicit edge rules: only gains -> scalar (100),
  only losses -> 0, both zero -> NaN (undefined). The ``0/0`` rule is
  applied LAST so it wins over the 100/0 rules in the ``np.where`` chain;
- TA-Lib is used only when its semantics match exactly (drift == 1,
  scalar == 100, length >= 2, all-finite input); otherwise the native
  path runs silently.
"""

import numpy as np
import polars as pl

from numba import float64, int64, njit

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available
from ..overlap.rma import rma_ind


# ----------------------------------------------------------------------
# Numba-accelerated gain/loss calculation
# ----------------------------------------------------------------------
@njit((float64[:], int64), fastmath=False, cache=True)
def _compute_gain_loss_numba(
    close: np.ndarray,
    drift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gain and loss arrays from close prices with given drift.

    IEEE 754: an undefined difference (NaN) propagates to both arrays;
    it is NOT silently treated as "no change".
    """
    n = len(close)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(drift, n):
        diff = close[i] - close[i - drift]
        if np.isnan(diff):
            gain[i] = np.nan
            loss[i] = np.nan
        elif diff > 0.0:
            gain[i] = diff
        elif diff < 0.0:
            loss[i] = -diff
        # else: exact zero -> both stay zero (flat bar)
    return gain, loss


# ----------------------------------------------------------------------
# Numpy-based RSI calculation
# ----------------------------------------------------------------------
def rsi_numpy(
    close: np.ndarray,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> np.ndarray:
    """Numpy-based RSI calculation with NaN handling and trim option.

    Parameters
    ----------
    close : np.ndarray
        Close prices (1-D). At least ``length + drift`` values are required.
    length : int
        RSI period (>= 1). TA-Lib path requires >= 2, otherwise the native
        path is used automatically.
    scalar : float
        Scaling factor (typically 100). TA-Lib path is used only for the
        canonical ``scalar == 100.0``.
    drift : int
        Lookback period for price differences (>= 1). TA-Lib path is used
        only for ``drift == 1``.
    offset : int
        Shift of the output series.
    fillna : float, optional
        Replacement for warm-up/shifted-in NaNs.
    use_talib : bool
        Prefer TA-Lib when its semantics match the requested parameters.
    nan_policy : str
        'raise' | 'ignore' | 'ffill' | 'bfill' | 'both'.
    trim : bool
        If True, drop the first ``length - 1`` (warm-up) values.
        NOTE: TA-Lib's own warm-up is one bar longer (``length`` values),
        so with the TA-Lib path a single leading NaN may remain after trim.

    Returns
    -------
    np.ndarray
        RSI values, same length as ``close`` (unless trimmed).

    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("RSI length must be >= 1")
    if drift < 1:
        raise ValueError("drift must be >= 1")

    close = np.asarray(close, dtype=np.float64)
    if close.ndim != 1:
        raise ValueError("close must be a 1-dimensional array")
    # Numba kernels require WRITABLE C-contiguous arrays. Polars'
    # zero-copy to_numpy() may return a read-only view, and
    # np.ascontiguousarray does NOT restore writeability on an
    # already-contiguous buffer -- force a real copy when needed.
    if not (close.flags.c_contiguous and close.flags.writeable):
        close = np.array(close, dtype=np.float64, order="C", copy=True)

    n = close.shape[0]
    if n < length + drift:
        raise ValueError(
            f"Input series too short: got {n} values, "
            f"need at least length + drift = {length + drift}."
        )

    # ---- IEEE 754: +/-Inf -> NaN (on a copy; the input is never mutated).
    # After conversion, NaN handling is delegated to nan_policy.
    if np.isinf(close).any():
        close = replace_inf_with_nan(close.copy())

    close = _handle_nan_policy(close, nan_policy, "close")

    # ---- Choose the code path ----
    # TA-Lib is used only when its semantics match exactly: drift == 1,
    # canonical scalar, timeperiod >= 2 and an all-finite input (TA-Lib's
    # NaN handling is version-dependent, so NaN input always takes the
    # native path where the NaN contract is enforced).
    use_native = use_talib and talib_available
    if use_native and (
        drift != 1 or scalar != 100.0 or length < 2 or np.isnan(close).any()  # noqa: RUF069 - exact IEEE zero/sign check
    ):
        use_native = False

    if use_native:
        rsi = talib.RSI(close, timeperiod=length)
    else:
        gain, loss = _compute_gain_loss_numba(close, drift)
        avg_gain = rma_ind(
            gain, length, offset=0, fillna=None, nan_policy=nan_policy
        )
        avg_loss = rma_ind(
            loss, length, offset=0, fillna=None, nan_policy=nan_policy
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss  # x/0 -> +inf, 0/0 -> NaN
            rsi = scalar - scalar / (1.0 + rs)  # +inf -> scalar, NaN -> NaN

        # Edge rules. ORDER MATTERS: for a flat market all three conditions
        # are true at once, and in a np.where chain the LAST rule wins.
        # The 0/0 rule must therefore come last:
        #   only gains -> scalar, only losses -> 0, both zero -> NaN.
        rsi = np.where(avg_loss == 0.0, scalar, rsi)  # noqa: RUF069 - exact IEEE zero/sign check
        rsi = np.where(avg_gain == 0.0, 0.0, rsi)  # noqa: RUF069 - exact IEEE zero/sign check
        rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), np.nan, rsi)  # noqa: RUF069 - exact IEEE zero/sign check

        # Strict NaN propagation: diffs touching a NaN close are undefined
        # and can never produce a value.
        rsi[np.isnan(gain)] = np.nan

    # ---- Trim (drops the first length-1 warm-up values) ----
    if trim:
        if len(rsi) >= length:
            rsi = rsi[length - 1 :]
        else:
            rsi = np.array([], dtype=np.float64)

    return _apply_offset_fillna(rsi, offset, fillna)


# ----------------------------------------------------------------------
# Universal RSI (accepts numpy array or Polars Series)
# ----------------------------------------------------------------------
def rsi_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> np.ndarray:
    """Universal RSI (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return rsi_numpy(
        close,
        length=length,
        scalar=scalar,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def rsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add RSI column to Polars
    DataFrame (default col name: ``RSI_{length}``).
    """
    # cast guarantees float64 output and null -> NaN conversion for int columns
    close = df[close_col].cast(pl.Float64).to_numpy()
    result = rsi_numpy(
        close,
        length=length,
        scalar=scalar,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    out_name = output_col or f"RSI_{length}"
    return df.with_columns(pl.Series(out_name, result))
