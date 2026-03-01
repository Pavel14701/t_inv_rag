import numpy as np
from numba import jit


# ----------------------------------------------------------------------
# Optimized offset and fillna in one Numba loop
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _apply_offset_fillna(
    arr: np.ndarray, 
    offset: int, 
    fillna: float | None
) -> np.ndarray:
    """
    Apply shift (offset) and fill NaN values in a single pass.

    Parameters
    ----------
    arr : np.ndarray
        Input array (float64).
    offset : int
        Shift amount. Positive = shift forward, negative = shift backward.
    fillna : float or None
        Value to replace NaNs. If None, NaNs remain.

    Returns
    -------
    np.ndarray
        New array with applied offset and fillna.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if offset > 0:
        # Shift forward: first 'offset' become NaN or fillna
        if fillna is not None:
            out[:offset] = fillna
        else:
            out[:offset] = np.nan
        out[offset:] = arr[:-offset]
    elif offset < 0:
        # Shift backward: last '|offset|' become NaN or fillna
        if fillna is not None:
            out[offset:] = fillna
        else:
            out[offset:] = np.nan
        out[:offset] = arr[-offset:]
    else:
        # No shift: just copy
        out[:] = arr
    # Fill any remaining NaNs (if fillna is given)
    if fillna is not None:
        for i in range(n):
            if np.isnan(out[i]):
                out[i] = fillna
    return out
