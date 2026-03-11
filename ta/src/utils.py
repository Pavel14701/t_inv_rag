import numpy as np
from numba import njit, types


@njit(
    (
        types.float64[:], 
        types.int64, 
        types.optional(types.float64)
    ), 
    cache=True, 
    fastmath=True
)
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
        Value to replace NaNs. If None, NaNs remain (but shifted positions become NaN).

    Returns
    -------
    np.ndarray
        New array with applied offset and fillna.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    fill_val = fillna if fillna is not None else np.nan
    if offset > 0:
        for i in range(offset):
            out[i] = fill_val
        for i in range(offset, n):
            v = arr[i - offset]
            if fillna is not None and np.isnan(v):
                out[i] = fillna
            else:
                out[i] = v
    elif offset < 0:
        off = -offset
        for i in range(n - off, n):
            out[i] = fill_val
        for i in range(n - off):
            v = arr[i + off]
            if fillna is not None and np.isnan(v):
                out[i] = fillna
            else:
                out[i] = v
    else:  # offset == 0
        for i in range(n):
            v = arr[i]
            if fillna is not None and np.isnan(v):
                out[i] = fillna
            else:
                out[i] = v
    return out


@njit((types.float64[:], types.int64), fastmath=True, cache=True)
def _rolling_min_numba(arr, window):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    dq = np.empty(window, dtype=np.int64)
    head = 0
    tail = 0
    size = 0
    for i in range(n):
        while size > 0:
            last_idx = dq[(tail - 1) % window]
            if arr[last_idx] >= arr[i]:
                tail = (tail - 1) % window
                size -= 1
            else:
                break
        dq[tail] = i
        tail = (tail + 1) % window
        size += 1
        while size > 0:
            first_idx = dq[head]
            if first_idx <= i - window:
                head = (head + 1) % window
                size -= 1
            else:
                break
        if i >= window - 1:
            out[i] = arr[dq[head]]
    return out


@njit((types.float64[:], types.int64), fastmath=True, cache=True)
def _rolling_max_numba(arr, window):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    dq = np.empty(window, dtype=np.int64)
    head = 0
    tail = 0
    size = 0
    for i in range(n):
        while size > 0:
            last_idx = dq[(tail - 1) % window]
            if arr[last_idx] <= arr[i]:
                tail = (tail - 1) % window
                size -= 1
            else:
                break
        dq[tail] = i
        tail = (tail + 1) % window
        size += 1
        while size > 0:
            first_idx = dq[head]
            if first_idx <= i - window:
                head = (head + 1) % window
                size -= 1
            else:
                break
        if i >= window - 1:
            out[i] = arr[dq[head]]
    return out


@njit(types.void(types.float64[:], types.unicode_type), cache=True, fastmath=True)
def _fill_nan_policy_numba(arr: np.ndarray, nan_policy: str) -> None:
    """
    Modify arr in-place according to nan_policy ('ffill', 'bfill', or 'both').
    Assumes arr has at least one NaN and is a copy.
    """
    n = len(arr)
    if nan_policy in ('ffill', 'both'):
        for i in range(1, n):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
    if nan_policy in ('bfill', 'both'):
        for i in range(n - 2, -1, -1):
            if np.isnan(arr[i]):
                arr[i] = arr[i + 1]


def _handle_nan_policy(arr: np.ndarray, nan_policy: str, name: str) -> np.ndarray:
    """Apply nan_policy to a single array (creates copy if needed)."""
    if not np.isnan(arr).any():
        return arr
    if nan_policy == 'raise':
        raise ValueError(f"Input {name} contains NaN values. "
                         "Use nan_policy='ffill', 'bfill' or 'both'.")
    arr = arr.copy()
    _fill_nan_policy_numba(arr, nan_policy)
    return arr
