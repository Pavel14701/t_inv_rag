"""Numba-accelerated array operations for rolling windows, NaN handling, and offset shifts.

All floating‑point operations strictly follow IEEE 754 rules (no fastmath optimisations).
NaN and Inf propagate naturally, and no exceptions are raised for extreme values.
"""
import numpy as np

from numba import njit, types


# -----------------------------------------------------------------------------
# Offset & NaN filling
# -----------------------------------------------------------------------------

@njit(
    (types.float64[:], types.int64, types.optional(types.float64)),
    cache=True,
    fastmath=False,
)
def _apply_offset_fillna(
    arr: np.ndarray,
    offset: int,
    fillna: float | None,
) -> np.ndarray:
    """Apply a shift (offset) and optionally fill NaN values in a single pass.

    The function creates a new array where the data is shifted by `offset`
    positions. Positive offset shifts the data forward (past values move to later
    positions), negative offset shifts backward. The first `abs(offset)`
    positions (or last for negative offset) are filled with `fillna` (or NaN if
    None). Additionally, any NaN in the original array is replaced with `fillna`
    if provided.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 input array.
    offset : int
        Number of positions to shift. Positive = forward (future),
        negative = backward (past).
    fillna : float or None
        Value to use for shifted‑in positions and for replacing NaNs.
        If None, NaN is used for both.

    Returns
    -------
    np.ndarray
        New array with the applied shift and NaN replacements.

    Notes
    -----
    - Empty arrays return an empty array.
    - All operations are IEEE 754 compliant (NaN and Inf are handled gracefully).

    """
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)

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
    else:
        for i in range(n):
            v = arr[i]
            if fillna is not None and np.isnan(v):
                out[i] = fillna
            else:
                out[i] = v
    return out


# -----------------------------------------------------------------------------
# Rolling window functions
# -----------------------------------------------------------------------------

@njit((types.float64[:], types.int64), fastmath=False, cache=True)
def _rolling_max_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba‑accelerated rolling maximum.

    If any NaN appears in the window, the result for that window is NaN
    (IEEE 754 propagation).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 input array.
    window : int
        Window size (must be >= 1). If window > len(arr) or window <= 0,
        all elements become NaN.

    Returns
    -------
    np.ndarray
        Array of rolling maxima, same length as `arr`, with first `window-1`
        elements set to NaN (incomplete window).

    Notes
    -----
    - NaN values in the input cause the corresponding window result to be NaN.
    - Empty input returns an empty array.

    """
    n = len(arr)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64) if n > 0 else np.empty(0, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    dq = np.empty(window, dtype=np.int64)
    head = 0
    tail = 0
    size = 0

    for i in range(n):
        val = arr[i]

        # If current value is NaN, reset the queue and store only this index.
        if np.isnan(val):
            head = 0
            tail = 0
            size = 0
            dq[0] = i
            head = 0
            tail = 1
            size = 1
            if i >= window - 1:
                out[i] = np.nan
            continue

        # Remove indices that are out of the current window.
        while size > 0:
            idx = dq[head]
            if idx <= i - window:
                head = (head + 1) % window
                size -= 1
            else:
                break

        # Remove from tail indices whose value <= current (for max).
        while size > 0:
            idx = dq[(tail - 1) % window]
            if not np.isnan(arr[idx]) and arr[idx] <= val:
                tail = (tail - 1) % window
                size -= 1
            else:
                break

        # Add current index.
        dq[tail] = i
        tail = (tail + 1) % window
        size += 1

        if i >= window - 1:
            # The head of the queue contains the index of the maximum (non‑NaN).
            out[i] = arr[dq[head]]

    return out


@njit((types.float64[:], types.int64), fastmath=False, cache=True)
def _rolling_min_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba‑accelerated rolling minimum.

    If any NaN appears in the window, the result for that window is NaN
    (IEEE 754 propagation).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 input array.
    window : int
        Window size (must be >= 1). If window > len(arr) or window <= 0,
        all elements become NaN.

    Returns
    -------
    np.ndarray
        Array of rolling minima, same length as `arr`, with first `window-1`
        elements set to NaN (incomplete window).

    Notes
    -----
    - NaN values in the input cause the corresponding window result to be NaN.
    - Empty input returns an empty array.

    """
    n = len(arr)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64) if n > 0 else np.empty(0, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    dq = np.empty(window, dtype=np.int64)
    head = 0
    tail = 0
    size = 0

    for i in range(n):
        val = arr[i]

        # If current value is NaN, reset the queue and store only this index.
        if np.isnan(val):
            head = 0
            tail = 0
            size = 0
            dq[0] = i
            head = 0
            tail = 1
            size = 1
            if i >= window - 1:
                out[i] = np.nan
            continue

        # Remove indices that are out of the current window.
        while size > 0:
            idx = dq[head]
            if idx <= i - window:
                head = (head + 1) % window
                size -= 1
            else:
                break

        # Remove from tail indices whose value >= current (for min).
        while size > 0:
            idx = dq[(tail - 1) % window]
            if not np.isnan(arr[idx]) and arr[idx] >= val:
                tail = (tail - 1) % window
                size -= 1
            else:
                break

        # Add current index.
        dq[tail] = i
        tail = (tail + 1) % window
        size += 1

        if i >= window - 1:
            # The head of the queue contains the index of the minimum (non‑NaN).
            out[i] = arr[dq[head]]

    return out


# -----------------------------------------------------------------------------
# NaN handling
# -----------------------------------------------------------------------------

@njit(types.void(types.float64[:], types.unicode_type), cache=True, fastmath=False)
def _fill_nan_policy_numba(arr: np.ndarray, nan_policy: str) -> None:
    """In‑place forward/backward fill of NaN values according to policy.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array (assumed to be a copy, modified in‑place).
    nan_policy : str
        One of 'ffill', 'bfill', or 'both'. 'ffill' propagates last valid value
        forward; 'bfill' propagates next valid value backward; 'both' does both.

    Notes
    -----
    - If the entire array is NaN, it remains unchanged (no value to fill).
    - This function is Numba‑compiled for speed.

    """
    n = len(arr)
    if n == 0:
        return

    if nan_policy in ('ffill', 'both'):
        for i in range(1, n):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
    if nan_policy in ('bfill', 'both'):
        for i in range(n - 2, -1, -1):
            if np.isnan(arr[i]):
                arr[i] = arr[i + 1]


def _handle_nan_policy(arr: np.ndarray, nan_policy: str, name: str) -> np.ndarray:
    """Apply a NaN handling policy to an array, optionally raising an error.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    nan_policy : str
        One of 'raise', 'ignore', 'ffill', 'bfill', or 'both'.
        'ignore' returns the array unchanged (allows NaNs to propagate).
    name : str
        Name of the array for error messages.

    Returns
    -------
    np.ndarray
        Array with NaNs handled (a copy if modifications were made,
        otherwise the original).

    Raises
    ------
    ValueError
        If `nan_policy == 'raise'` and any NaN is present.

    """
    _VALID_POLICIES = ('raise', 'ignore', 'ffill', 'bfill', 'both')
    if nan_policy not in _VALID_POLICIES:
        raise ValueError(
            f'Unknown nan_policy: {nan_policy}. '
            "Use 'raise', 'ignore', 'ffill', 'bfill', or 'both'."
        )
    if nan_policy == 'ignore':
        return arr
    if not np.isnan(arr).any():
        return arr
    if nan_policy == 'raise':
        raise ValueError(
            f"Input {name} contains NaN values. "
            "Use nan_policy='ffill', 'bfill' or 'both'."
        )
    arr = arr.copy()
    _fill_nan_policy_numba(arr, nan_policy)
    return arr


def replace_inf_with_nan(arr: np.ndarray) -> np.ndarray:
    """Replace all infinite values (inf and -inf) with NaN in‑place.

    Parameters
    ----------
    arr : np.ndarray
        Input array (modified in‑place).

    Returns
    -------
    np.ndarray
        The same array with infinities replaced by NaN (convenience return).

    Notes
    -----
    - This operation is IEEE 754 compliant and does not raise errors.

    """
    arr[~np.isfinite(arr)] = np.nan
    return arr