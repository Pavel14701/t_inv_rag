# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl
from numba import float64, int64, njit

from .. import talib, talib_available
from ..ma import ma_mode
from ..utils import _apply_offset_fillna, _handle_nan_policy
from ..volatility import atr_ind


# ----------------------------------------------------------------------
# Helper for TradingView mode (tvmode=True) with Numba signature
# ----------------------------------------------------------------------
@njit(
    (float64[:], float64[:], int64, int64, float64, float64[:]), 
    cache=True, 
    fastmath=True
)
def _tv_dmp_dmn_adx(
    pos: np.ndarray,
    neg: np.ndarray,
    length: int,
    signal_length: int,
    scalar: float,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DMP, DMN and ADX according to TradingView logic.
    Returns (dmp, dmn, adx).
    """
    n = len(pos)
    k = scalar / atr
    dmp = np.empty(n, dtype=np.float64)
    dmn = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)
    adx = np.empty(n, dtype=np.float64)
    dmp[:length - 1] = np.nan
    dmn[:length - 1] = np.nan
    dx[:length - 1] = np.nan
    adx[:length - 1] = np.nan
    # Compute sum of first `length` raw pos and neg
    sum_pos = 0.0
    sum_neg = 0.0
    for i in range(length):
        sum_pos += pos[i]
        sum_neg += neg[i]
    dmp[length - 1] = k[length - 1] * sum_pos
    dmn[length - 1] = k[length - 1] * sum_neg
    alpha = 1.0 / length
    for i in range(length, n):
        dmp[i] = alpha * k[i] * pos[i] + (1.0 - alpha) * dmp[i - 1]
        dmn[i] = alpha * k[i] * neg[i] + (1.0 - alpha) * dmn[i - 1]
    for i in range(length - 1, n):
        denom = dmp[i] + dmn[i]
        if denom != 0.0:
            dx[i] = scalar * abs(dmp[i] - dmn[i]) / denom
        else:
            dx[i] = np.nan
    # TradingView shifts DX backward by `length` bars
    # dx_shifted[i] = dx[i + length]
    dx_shifted = np.full(n, np.nan, dtype=np.float64)
    if n > length:
        dx_shifted[:-length] = dx[length:]
    # ADX is RMA of the shifted DX with period signal_length
    adx_start_idx = length + signal_length - 1
    if adx_start_idx < n:
        # Initial SMA
        sum_dx = 0.0
        for i in range(length, length + signal_length):
            sum_dx += dx_shifted[i]
        adx[adx_start_idx] = sum_dx / signal_length
        alpha_sig = 1.0 / signal_length
        for i in range(adx_start_idx + 1, n):
            adx[i] = alpha_sig * dx_shifted[i] + (1.0 - alpha_sig) * adx[i - 1]
    return dmp, dmn, adx


# ----------------------------------------------------------------------
# Core ADX calculation with improvements
# ----------------------------------------------------------------------
def adx_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based ADX calculation with NaN handling and trim option.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays.
    length : int
        Period for ATR and directional movement (default 14).
    signal_length : int, optional
        Period for ADX smoothing (defaults to length).
    adxr_length : int
        Period for ADXR calculation (default 2).
    scalar : float
        Scaling factor (typically 100).
    tvmode : bool
        If True, use TradingView algorithm (different smoothing).
    mamode : str
        Moving average mode for standard calculation (ignored if tvmode=True).
    drift : int
        Lookback period for price differences.
    offset, fillna, use_talib : as usual.
    nan_policy : str, default 'raise'
        How to handle NaNs in input arrays ('raise', 'ffill', 'bfill', 'both').
    trim : bool, default False
        If True, return only the valid part of \
            all series (first `length+signal_length-1` values removed).
        Output length becomes `len(close) - (length + signal_length - 1)`.

    Returns
    -------
    (adx, adxr, dmp, dmn) as numpy arrays, possibly trimmed.
    """
    if signal_length is None:
        signal_length = length
    # ---- Input validation ----
    if length < 1 or signal_length < 1:
        raise ValueError("length and signal_length must be >= 1")
    if adxr_length < 1:
        raise ValueError("adxr_length must be >= 1")
    # Convert to float64 and check contiguity
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    # Check for infinite values
    for name, arr in [("high", high), ("low", low), ("close", close)]:
        if np.isinf(arr).any():
            raise ValueError(f"Input {name} contains non-finite values (inf or -inf).")
    # Apply NaN policy to each array
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure C-contiguous for performance (fixing the previous bug)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # 1. ATR
    atr = atr_ind(
        high, low, close,
        length=length,
        mamode="rma",
        drift=drift,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,   # assume atr_ind also supports nan_policy
    )
    if np.all(np.isnan(atr)):
        raise ValueError("ATR calculation failed")
    # 2. Up and Down movements
    up = high - np.roll(high, drift)
    up[:drift] = np.nan
    dn = np.roll(low, drift) - low
    dn[:drift] = np.nan
    # 3. Positive and negative directional movement
    up_gt_dn = up > dn
    dn_gt_up = dn > up
    pos = np.where(up_gt_dn & (up > 0), up, 0.0).astype(np.float64)
    neg = np.where(dn_gt_up & (dn > 0), dn, 0.0).astype(np.float64)
    # 4. Compute DMP, DMN, ADX
    if use_talib and talib_available and not tvmode:
        # TA‑Lib does not support tvmode
        dmp = talib.PLUS_DM(high, low, timeperiod=length)
        dmn = talib.MINUS_DM(high, low, timeperiod=length)
        adx = talib.ADX(high, low, close, timeperiod=length)
    else:
        if tvmode:
            dmp, dmn, adx = _tv_dmp_dmn_adx(
                pos, neg, length, signal_length, scalar, atr
            )
        else:
            # Standard calculation using MA of pos/neg
            k = scalar / atr
            dmp = k * cast(np.ndarray, ma_mode(
                mamode, pos, length=length, offset=0, fillna=None, use_talib=False
            ))
            dmn = k * cast(np.ndarray, ma_mode(
                mamode, neg, length=length, offset=0, fillna=None, use_talib=False
            ))
            denom = dmp + dmn
            with np.errstate(divide='ignore', invalid='ignore'):
                dx = scalar * np.abs(dmp - dmn) / denom
                dx = np.where(denom == 0.0, np.nan, dx)
            adx = cast(np.ndarray, ma_mode(
                mamode, dx, length=signal_length, offset=0, fillna=None, use_talib=False
            ))
    # 5. ADXR
    adx_shifted = np.roll(adx, adxr_length)
    adx_shifted[:adxr_length] = np.nan
    adxr = 0.5 * (adx + adx_shifted)
    # 6. Trim if requested
    if trim:
        start = length + signal_length - 1
        if start >= len(adx):
            raise ValueError("Trim start index exceeds array length. Series too short.")
        adx = adx[start:]
        adxr = adxr[start:]
        dmp = dmp[start:]
        dmn = dmn[start:]
    # 7. Apply offset and fillna
    adx = _apply_offset_fillna(adx, offset, fillna)
    adxr = _apply_offset_fillna(adxr, offset, fillna)
    dmp = _apply_offset_fillna(dmp, offset, fillna)
    dmn = _apply_offset_fillna(dmn, offset, fillna)
    return adx, adxr, dmp, dmn


def adx_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal ADX (accepts numpy arrays or Polars Series) with NaN handling and trim.
    Returns (adx, adxr, dmp, dmn) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return adx_numpy(
        high, low, close,
        length=length,
        signal_length=signal_length,
        adxr_length=adxr_length,
        scalar=scalar,
        tvmode=tvmode,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


def adx_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    suffix: str = "",
) -> pl.DataFrame:
    """
    ADX for Polars DataFrame (does not support trim because it must return same length).

    Columns added:
        ADX_{signal_length}
        ADXR_{signal_length}_{adxr_length}
        DMP_{length}
        DMN_{length}
    """
    if signal_length is None:
        signal_length = length
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    adx_arr, adxr_arr, dmp_arr, dmn_arr = adx_numpy(
        high, low, close,
        length=length,
        signal_length=signal_length,
        adxr_length=adxr_length,
        scalar=scalar,
        tvmode=tvmode,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    suffix = suffix or f"_{signal_length}"
    return pl.DataFrame({
        date_col: df[date_col],
        f"ADX{suffix}": adx_arr,
        f"ADXR_{signal_length}_{adxr_length}": adxr_arr,
        f"DMP_{length}": dmp_arr,
        f"DMN_{length}": dmn_arr,
    })