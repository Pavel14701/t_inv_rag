import numpy as np
from numba import jit

from ..overlap import sma_ind
from ..volume import vwma_ind


# ----------------------------------------------------------------------
# Core Numba functions
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _price_v_rolling(
    price: np.ndarray,
    vpr: np.ndarray,
    lenV: np.ndarray,
    VPCc: np.ndarray,
) -> np.ndarray:
    """
    Rolling average of price / (VPCc * vpr) with dynamic window length.
    """
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        L = lenV[i]
        if L > 0:
            start = max(0, i - L + 1)
            denom = VPCc[i] * vpr[start:i + 1]
            valid = (VPCc[i] != 0) & (vpr[start:i + 1] != 0)
            values = np.divide(
                price[start:i + 1],
                denom,
                out=np.zeros_like(price[start:i + 1]),
                where=valid
            )
            out[i] = np.sum(values) / L / 100.0
        else:
            out[i] = price[i]
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _compute_len_v(vpc: np.ndarray, vpci: np.ndarray) -> np.ndarray:
    """Dynamic window length based on VPCI and VPC."""
    n = len(vpc)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        if np.isnan(vpci[i]):
            out[i] = 1
        elif vpc[i] < 0:
            out[i] = int(round(abs(vpci[i] - 3)))
        else:
            out[i] = int(round(vpci[i] + 3))
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _compute_vpcc(vpc: np.ndarray) -> np.ndarray:
    """Clamp VPC to avoid near‑zero values."""
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
    """
    Compute common series for AVSL/AVSR:
        vpc, vpr, vm, vpci, deviation_raw
    """
    # Volume‑weighted and simple moving averages
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

