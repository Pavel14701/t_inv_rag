# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, njit

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Core SCRSI calculations with Numba acceleration
# ----------------------------------------------------------------------
@njit((float64[:], int64, int64), fastmath=True, cache=True)
def _cyclic_smoothing(rsi_scaled: np.ndarray, vibration: int) -> np.ndarray:
    """
    Apply cyclic smoothing to scaled RSI.

    Parameters
    ----------
    rsi_scaled : np.ndarray
        RSI scaled to [-100, 100].
    vibration : int
        Vibration parameter controlling smoothing.

    Returns
    -------
    np.ndarray
        Smoothed SCRSI values.
    """
    n = len(rsi_scaled)
    torque = 2.0 / (vibration + 1.0)
    phasing_lag = (vibration - 1) // 2
    crsi = np.zeros(n, dtype=np.float64)
    for i in range(phasing_lag, n):
        crsi[i] = torque * (
            2.0 * rsi_scaled[i] - rsi_scaled[i - phasing_lag]
        ) + (1.0 - torque) * crsi[i - 1]
    return crsi


def scrsi_numpy(
    close: np.ndarray,
    domcycle: int,
    vibration: int,
    leveling: float,
    nan_policy: str = 'raise',
    trim: bool = False,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Numpy‑based Smooth Cicle RSI calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    domcycle : int
        Dominant cycle length (must be >= 2).
    vibration : int
        Vibration parameter (must be >= 2).
    leveling : float
        Percentile level for boundaries (0 < leveling < 50).
    nan_policy : str, default 'raise'
        How to handle NaNs in input: 'raise', 'ffill', 'bfill', 'both'.
    trim : bool, default False
        If True, return only valid \
            part (first `domcycle//2 + (vibration-1)//2 -1` values removed).
    offset, fillna : as usual.

    Returns
    -------
    (rsi_scaled, crsi, lower_bound, upper_bound)
    """
    # Input validation
    if domcycle < 2:
        raise ValueError("domcycle must be >= 2")
    if vibration < 2:
        raise ValueError("vibration must be >= 2")
    if not (0 < leveling < 50):
        raise ValueError("leveling must be between 0 and 50")
    if vibration > domcycle:
        raise ValueError("vibration should not exceed domcycle.")
    close = np.asarray(close, dtype=np.float64)
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    close = _handle_nan_policy(close, nan_policy, "close")
    # Check if series is long enough
    min_length = domcycle + vibration
    if len(close) < min_length:
        raise ValueError(
            f"Input series too short: need at least \
                {min_length} elements, got {len(close)}."
        )
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    cyclelen = domcycle // 2
    cyclicmemory = domcycle * 2
    # Price differences (drift=1) – faster version
    diff = close - np.roll(close, 1)
    diff[0] = 0.0
    up_raw = np.maximum(diff, 0.0)
    down_raw = np.maximum(-diff, 0.0)
    # Rolling averages (SMA)
    up = sma_ind(up_raw, length=cyclelen, use_talib=False, nan_policy=nan_policy)
    down = sma_ind(down_raw, length=cyclelen, use_talib=False, nan_policy=nan_policy)
    # RSI calculation with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = up / down
        rsi = 100.0 - 100.0 / (1.0 + rs)
        rsi = np.where(down == 0.0, 100.0, rsi)          # if down=0, RSI=100
        rsi = np.where(up == 0.0, 0.0, rsi)              # if up=0, RSI=0
        # When both zero, set to 50 (neutral) instead of NaN for stability
        rsi = np.where((up == 0.0) & (down == 0.0), 50.0, rsi)
    rsi = np.clip(rsi, 0.0, 100.0)
    # Scale to [-100, 100]
    rsi_scaled = (rsi - 50.0) * 2.0
    # Cyclic smoothing
    crsi = _cyclic_smoothing(rsi_scaled, vibration)
    # Boundaries (first cyclicmemory values after removing NaNs)
    crsi_clean = crsi[~np.isnan(crsi)]
    if len(crsi_clean) >= cyclicmemory:
        lower_bound = float(np.percentile(crsi_clean[:cyclicmemory], leveling))
        upper_bound = float(np.percentile(crsi_clean[:cyclicmemory], 100.0 - leveling))
    else:
        lower_bound = -100.0
        upper_bound = 100.0
    # Trim if requested
    if trim:
        phasing_lag = (vibration - 1) // 2
        start = cyclelen + phasing_lag - 1
        if start < len(crsi):
            rsi_scaled = rsi_scaled[start:]
            crsi = crsi[start:]
        else:
            rsi_scaled = np.array([])
            crsi = np.array([])
    # Apply offset and fillna
    rsi_scaled = _apply_offset_fillna(rsi_scaled, offset, fillna)
    crsi = _apply_offset_fillna(crsi, offset, fillna)
    return rsi_scaled, crsi, lower_bound, upper_bound


def scrsi_ind(
    close: np.ndarray | pl.Series,
    domcycle: int,
    vibration: int,
    leveling: float,
    nan_policy: str = 'raise',
    trim: bool = False,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Universal SCRSI (accepts numpy array or Polars Series).
    Returns (rsi_scaled, crsi, lower_bound, upper_bound).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return scrsi_numpy(
        close,
        domcycle=domcycle,
        vibration=vibration,
        leveling=leveling,
        nan_policy=nan_policy,
        trim=trim,
        offset=offset,
        fillna=fillna,
    )


def scrsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    domcycle: int = 14,
    vibration: int = 5,
    leveling: float = 10.0,
    nan_policy: str = 'raise',
    offset: int = 0,
    fillna: float | None = None,
    output_col_scaled: str | None = None,
    output_col_crsi: str | None = None,
    output_col_lb: str | None = None,
    output_col_ub: str | None = None,
) -> pl.DataFrame:
    """
    Add SCRSI columns to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Column with close prices.
    domcycle, vibration, leveling, nan_policy, offset, fillna : as in scrsi_numpy.
    output_col_scaled : str, optional
        Name for scaled RSI column (default f"SCRSI_scaled_{domcycle}_{vibration}").
    output_col_crsi : str, optional
        Name for smoothed CRSI column (default f"SCRSI_crsi_{domcycle}_{vibration}").
    output_col_lb : str, optional
        Name for lower bound column (default f"SCRSI_lb_{domcycle}_{vibration}").
    output_col_ub : str, optional
        Name for upper bound column (default f"SCRSI_ub_{domcycle}_{vibration}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added columns (same length).
    """
    close = df[close_col].to_numpy()
    rsi_scaled, crsi, lb, ub = scrsi_numpy(
        close,
        domcycle=domcycle,
        vibration=vibration,
        leveling=leveling,
        nan_policy=nan_policy,
        trim=False,      # Polars всегда возвращает полную длину
        offset=offset,
        fillna=fillna,
    )
    # Default column names
    suffix = f"_{domcycle}_{vibration}"
    if output_col_scaled is None:
        output_col_scaled = f"SCRSI_scaled{suffix}"
    if output_col_crsi is None:
        output_col_crsi = f"SCRSI_crsi{suffix}"
    if output_col_lb is None:
        output_col_lb = f"SCRSI_lb{suffix}"
    if output_col_ub is None:
        output_col_ub = f"SCRSI_ub{suffix}"
    return df.with_columns([
        pl.Series(output_col_scaled, rsi_scaled),
        pl.Series(output_col_crsi, crsi),
        pl.Series(output_col_lb, [lb] * len(df)),   # константа для всех строк
        pl.Series(output_col_ub, [ub] * len(df)),
    ])