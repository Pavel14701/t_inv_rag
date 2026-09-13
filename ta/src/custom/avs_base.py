import numpy as np

from numba import jit

from ..overlap.sma import sma_ind
from ..volume.vwma import vwma_ind


# ----------------------------------------------------------------------
# Core Numba functions (IEEE-754 compliant)
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _price_v_rolling(
    price: np.ndarray,
    vpr: np.ndarray,
    len_v: np.ndarray,
    vpc_c: np.ndarray,
) -> np.ndarray:
    """Compute the rolling average of price / (VPCc * vpr)
    with a dynamic window length.

    For each index i, the window length L = len_v[i]. If L > 0,
    the window spans from start = max(0, i-L+1) to i inclusive. Values where
    the denominator is zero are ignored (replaced with 0). The result is
    normalised by dividing by L and by 100.

    Parameters
    ----------
    price : np.ndarray, shape (n,), dtype=np.float64
        Prices (e.g., low or high).
    vpr : np.ndarray, shape (n,), dtype=np.float64
        Volume Price Ratio (VWMA_fast / SMA_fast).
    len_v : np.ndarray, shape (n,), dtype=np.int32
        Dynamic window length per index (computed by _compute_len_v).
    vpc_c : np.ndarray, shape (n,), dtype=np.float64
        Clamped VPC to avoid division by zero (values in [-1,0) and [1,+inf)).

    Returns
    -------
    np.ndarray, shape (n,), dtype=np.float64
        Rolling averages normalised by 100. For L=0, returns price[i].

    Notes
    -----
    This function is compiled with Numba (nopython mode) and follows IEEE 754.
    Division by zero is handled via np.divide with a 'where' mask.

    """
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        L = len_v[i]  # noqa: N806
        if L > 0:
            start = max(0, i - L + 1)
            denom = vpc_c[i] * vpr[start : i + 1]
            valid = (vpc_c[i] != 0) & (vpr[start : i + 1] != 0)
            values = np.divide(
                price[start : i + 1],
                denom,
                out=np.zeros_like(price[start : i + 1]),
                where=valid,
            )
            out[i] = np.sum(values) / L / 100.0
        else:
            out[i] = price[i]
    return out


@jit(nopython=True, cache=True)
def _compute_len_v(vpc: np.ndarray, vpci: np.ndarray) -> np.ndarray:
    """Compute dynamic window length based on VPC and VPCI
    (Volume Price Confirmation Indicator).

    If VPCI contains NaN, length = 1.
    Else, if VPC < 0, length = round(abs(VPCI - 3)),
    otherwise length = round(VPCI + 3).

    Parameters
    ----------
    vpc : np.ndarray, shape (n,), dtype=np.float64
        Volume Price Criterion (VWMA_slow - SMA_slow).
    vpci : np.ndarray, shape (n,), dtype=np.float64
        Volume Price Confirmation Indicator (VPC * VPR * VM).

    Returns
    -------
    np.ndarray, shape (n,), dtype=np.int32
        Window length for each index (minimum 1).

    Notes
    -----
    Uses Python's round() (bankers' rounding) - standard Python behaviour.

    """
    n = len(vpc)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        if np.isnan(vpci[i]):
            out[i] = 1
        elif vpc[i] < 0:
            out[i] = round(abs(vpci[i] - 3))
        else:
            out[i] = round(vpci[i] + 3)
    return out


@jit(nopython=True, cache=True)
def _compute_vpcc(vpc: np.ndarray) -> np.ndarray:
    """Clamp VPC to avoid division by zero and extremely small values.

    Transformations:
    - if -1.0 < val < 0.0 -> -1.0
    - if 0.0 <= val < 1.0 -> 1.0
    - otherwise val remains unchanged.

    Parameters
    ----------
    vpc : np.ndarray, shape (n,), dtype=np.float64
        Original Volume Price Criterion.

    Returns
    -------
    np.ndarray, shape (n,), dtype=np.float64
        Clamped VPC array.

    Notes
    -----
    Ensures that the denominator in _price_v_rolling is
    never too close to zero.

    """
    out = np.empty_like(vpc)
    for i in range(len(vpc)):
        val = vpc[i]
        if val > -1.0 and val < 0.0:
            out[i] = -1.0
        elif val >= 0.0 and val < 1.0:
            out[i] = 1.0
        else:
            out[i] = val
    return out


# ----------------------------------------------------------------------
# Common AVS calculation (shared between support and resistance)
# ----------------------------------------------------------------------
def _avs_base(
    close: np.ndarray,
    volume: np.ndarray,
    fast: int,
    slow: int,
    stand_div: float,
    use_talib: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the common series required for both AVSL (support)
    and AVSR (resistance).

    Based on closing prices and volumes, it calculates fast and slow VWMA,
    as well as simple moving averages of prices and volumes.
    The derived series are:
    VPC = VWMA_slow - SMA_slow
    VPR = VWMA_fast / SMA_fast
    VM  = SMA_vol_fast / SMA_vol_slow
    VPCI = VPC * VPR * VM
    deviation_raw = stand_div * VPCI * VM

    Parameters
    ----------
    close : np.ndarray, shape (n,), dtype=np.float64
        Closing prices.
    volume : np.ndarray, shape (n,), dtype=np.float64
        Trading volumes.
    fast : int
        Fast period length (e.g., 5).
    slow : int
        Slow period length (e.g., 20).
    stand_div : float
        Multiplier for the deviation calculation.
    use_talib : bool
        If True, uses TA-Lib for moving averages (usually faster);
        otherwise uses a custom implementation.

    Returns
    -------
    tuple (vpc, vpr, vm, vpci, deviation_raw)
        vpc : np.ndarray, shape (n,)
            Volume Price Criterion.
        vpr : np.ndarray, shape (n,)
            Volume Price Ratio.
        vm : np.ndarray, shape (n,)
            Volume Ratio (fast volume MA / slow volume MA).
        vpci : np.ndarray, shape (n,)
            Volume Price Confirmation Indicator.
        deviation_raw : np.ndarray, shape (n,)
            Unclamped deviation value (before applying max_deviation).

    Notes
    -----
    This function uses vectorised operations (no explicit loops).
    All computations are performed in float64.

    """
    # Volume-weighted and simple moving averages
    vwma_fast = vwma_ind(close, volume, fast, use_talib=use_talib)
    vwma_slow = vwma_ind(close, volume, slow, use_talib=use_talib)
    sma_fast = sma_ind(close, fast, use_talib=use_talib)
    sma_slow = sma_ind(close, slow, use_talib=use_talib)
    vol_fast = sma_ind(volume, fast, use_talib=use_talib)
    vol_slow = sma_ind(volume, slow, use_talib=use_talib)
    # Derived series
    vpc = vwma_slow - sma_slow
    vpr = vwma_fast / sma_fast
    vm = vol_fast / vol_slow
    vpci = vpc * vpr * vm
    deviation_raw = stand_div * vpci * vm
    return vpc, vpr, vm, vpci, deviation_raw
