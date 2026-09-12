# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from numba import float64, int64, njit  # type: ignore[attr-defined]
from numba.typed import List


@njit(
    (
        float64[:],   # x
        float64,      # prominence
        int64,        # distance
        int64,        # plateau_size (-1 = off)
        float64,      # rel_height
        float64,      # width (-1 = off)
        int64         # wlen (-1 = off)
    ),
    fastmath=False,  # strict IEEE 754; fastmath gave no speed-up here
                     # (branch-heavy kernel) and its nnan/ninf/contract
                     # flags could alter the `x[i] <= h` width boundary
    cache=True
)
def _find_peaks_nb(x, prominence, distance, plateau_size, rel_height, width, wlen):
    n = len(x)
    peaks = List.empty_list(int64)

    # ---- Step 1: local maxima incl. flat tops (plateaus) ----
    # Walk rising slopes; a run [i, j] of equal values is a peak iff the
    # value right of the run is strictly lower (the left side is already
    # strictly lower by the loop guard).  The peak index is the midpoint
    # of the run (scipy convention).  A run of length 1 is a plain peak.
    i = 1
    while i < n - 1:
        if x[i] > x[i - 1]:
            j = i
            while j < n - 1 and x[j + 1] == x[i]:
                j += 1
            # Right side must fall strictly for the run to be a maximum
            if j < n - 1 and x[j + 1] < x[i]:
                run_len = j - i + 1
                if plateau_size < 0 or run_len >= plateau_size:
                    peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    # scipy evaluates filters sequentially in the order
    # plateau_size -> height -> threshold -> distance -> prominence ->
    # width, and prominence is computed only for peaks that survived the
    # previous filters.  So the distance greedy runs FIRST, over ALL
    # local maxima, with the peak height as priority.
    # ---- Step 2: distance filter (scipy-style greedy) ----
    # Repeatedly keep the highest remaining peak and drop every peak
    # closer than `distance` bars to it, so all surviving peaks are at
    # least `distance` apart.
    if distance > 1 and len(peaks) > 1:
        m = len(peaks)
        idx = np.empty(m, dtype=np.int64)
        for i in range(m):
            idx[i] = peaks[i]
        active = np.ones(m, dtype=np.bool_)
        kept = List.empty_list(int64)
        while True:
            best = -1
            for i in range(m):
                if active[i] and (best < 0 or x[idx[i]] > x[idx[best]]):
                    best = i
            if best < 0:
                break
            kept.append(idx[best])
            for i in range(m):
                if active[i] and abs(idx[i] - idx[best]) < distance:
                    active[i] = False
        peaks = kept
    # ---- Step 3: prominence and width filters ----
    if prominence > 0 or width >= 0:
        filtered = List.empty_list(int64)
        for idx in range(len(peaks)):
            p = peaks[idx]
            peak_val = x[p]
            # Determine search window (wlen)
            if wlen > 0:
                half = wlen // 2
                left_bound = p - half
                if left_bound < 0:
                    left_bound = 0
                right_bound = p + half
                if right_bound > n - 1:
                    right_bound = n - 1
            else:
                left_bound = 0
                right_bound = n - 1
            # Left base – scan outward until the first strictly higher
            # sample (the "col" toward a higher peak, scipy definition)
            # or the window border; track the minimum on the way.
            left_min = peak_val
            for j in range(p - 1, left_bound - 1, -1):
                if x[j] > peak_val:
                    break
                if x[j] < left_min:
                    left_min = x[j]
            # Right base – symmetric
            right_min = peak_val
            for j in range(p + 1, right_bound + 1):
                if x[j] > peak_val:
                    break
                if x[j] < right_min:
                    right_min = x[j]
            prom = peak_val - max(left_min, right_min)
            if prom < prominence:
                continue
            # Width at rel_height
            if width >= 0:
                h = peak_val - prom * rel_height
                # Find left intersection (closest to p)
                wl = p
                for j in range(p, left_bound - 1, -1):
                    if x[j] <= h:
                        wl = j
                        break
                # Find right intersection (closest to p)
                wr = p
                for j in range(p, right_bound + 1):
                    if x[j] <= h:
                        wr = j
                        break
                w = wr - wl
                if w < width:
                    continue
            filtered.append(p)
        peaks = filtered
    # Convert to numpy array and sort (just in case)
    out = np.empty(len(peaks), dtype=np.int64)
    for i in range(len(peaks)):
        out[i] = peaks[i]
    out.sort()
    return out


def zigzag_peaks_valleys(
    high: np.ndarray,
    low: np.ndarray,
    prominence_peak: float,
    prominence_valley: float,
    distance: int,
    width: float | None,
    wlen: int | None,
    rel_height: float,
    plateau_size: int | None,
):
    plateau = plateau_size if plateau_size is not None else -1
    width_ = width if width is not None else -1.0
    wlen_ = wlen if wlen is not None else -1
    peaks = _find_peaks_nb(
        high,
        prominence_peak,
        distance,
        plateau,
        rel_height,
        width_,
        wlen_,
    )
    valleys = _find_peaks_nb(
        -low,
        prominence_valley,
        distance,
        plateau,
        rel_height,
        width_,
        wlen_,
    )
    return peaks, valleys


# ----------------------------------------------------------------------
# Public Numpy function (returns peak and valley indices)
# ----------------------------------------------------------------------
def zigzag_numpy(
    high: np.ndarray,
    low: np.ndarray,
    prominence_peak: float = 0.01,
    prominence_valley: float = 0.01,
    distance: int = 5,
    width: float | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks (in high) and valleys (in low) using custom Numba peak detection.

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays (float64).
    prominence_peak, prominence_valley : float
        Minimum prominence for peaks and valleys.
    distance : int
        Minimum number of samples between neighbouring peaks.
    width : float, optional
        Required width of peaks.
    wlen : int, optional
        Window length for prominence calculation.
    rel_height : float
        Relative height at which the width is measured (0 < rel_height ≤ 1).
    plateau_size : int, optional
        Minimum plateau length.

    Returns
    -------
    peak_indices : np.ndarray
        Indices of detected peaks in the high series.
    valley_indices : np.ndarray
        Indices of detected valleys in the low series.

    """
    # Input validation
    if np.any(np.isnan(high)):
        raise ValueError('high array contains NaNs')
    if np.any(np.isnan(low)):
        raise ValueError('low array contains NaNs')
    if np.any(np.isinf(high)) or np.any(np.isinf(low)):
        raise ValueError('high/low arrays contain Inf values')
    if len(high) != len(low):
        raise ValueError(
            'high and low must have the same length: '
            f'got {len(high)} and {len(low)}.'
        )
    # Convert optional parameters to sentinel values expected by Numba
    plateau = plateau_size if plateau_size is not None else -1
    width_ = width if width is not None else -1.0
    wlen_ = wlen if wlen is not None else -1
    # Ensure arrays are float64, C-contiguous and writable
    # (pl.Series.to_numpy() may return a read-only view)
    high = np.require(high, dtype=np.float64, requirements=['C', 'W'])
    low = np.require(low, dtype=np.float64, requirements=['C', 'W'])
    peaks = _find_peaks_nb(
        high,
        prominence_peak,
        distance,
        plateau,
        rel_height,
        width_,
        wlen_,
    )
    valleys = _find_peaks_nb(
        -low,                      # invert low to find valleys as peaks
        prominence_valley,
        distance,
        plateau,
        rel_height,
        width_,
        wlen_,
    )
    return peaks, valleys


# ----------------------------------------------------------------------
# Universal wrapper (accepts numpy arrays or Polars Series)
# ----------------------------------------------------------------------
def zigzag_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    prominence_peak: float = 0.01,
    prominence_valley: float = 0.01,
    distance: int = 5,
    width: float | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal Zigzag indicator (returns numpy arrays of indices)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    return zigzag_numpy(
        high, low,
        prominence_peak=prominence_peak,
        prominence_valley=prominence_valley,
        distance=distance,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def zigzag_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    prominence_peak: float = 0.01,
    prominence_valley: float = 0.01,
    distance: int = 5,
    width: float | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | None = None,
    suffix: str = '',
) -> pl.DataFrame:
    """Add boolean columns 'is_peak' and 'is_valley' to the Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col : str
        Names of the columns containing high and low prices.
    prominence_peak, prominence_valley, distance, \
        width, wlen, rel_height, plateau_size :
        Parameters for peak/valley detection (see zigzag_numpy).
    suffix : str
        Optional suffix for the new columns (e.g., "_zz").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with two added boolean columns:
        'is_peak{suffix}' and 'is_valley{suffix}'.  The input
        DataFrame is not modified in place.

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()

    peak_idx, valley_idx = zigzag_numpy(
        high, low,
        prominence_peak=prominence_peak,
        prominence_valley=prominence_valley,
        distance=distance,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )
    # Create boolean masks
    is_peak = np.zeros(len(df), dtype=bool)
    is_valley = np.zeros(len(df), dtype=bool)
    is_peak[peak_idx] = True
    is_valley[valley_idx] = True
    return df.with_columns([
        pl.Series(f'is_peak{suffix}', is_peak),
        pl.Series(f'is_valley{suffix}', is_valley),
    ])