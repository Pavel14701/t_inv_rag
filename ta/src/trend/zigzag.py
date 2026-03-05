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
    nopython=True,
    fastmath=True,
    cache=True
)
def _find_peaks_nb(x, prominence, distance, plateau_size, rel_height, width, wlen):
    n = len(x)
    peaks = List.empty_list(int64)

    # ---- Step 1: local maxima + plateaus (as single peaks) ----
    i = 1
    while i < n - 1:
        # Single peak
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(i)
            i += 1
            continue

        # Plateau: flat region of at least plateau_size
        if plateau_size >= 0 and x[i] == x[i - 1] and x[i] == x[i + 1]:
            left = i
            while left > 0 and x[left] == x[i]:
                left -= 1
            right = i
            while right < n - 1 and x[right] == x[i]:
                right += 1
            length = right - left - 1
            if length >= plateau_size:
                # Check if plateau is a local maximum: 
                # values left of left and right of right must be lower
                left_ok = (left == 0) or (x[left] > x[left - 1])
                right_ok = (right == n - 1) or (x[right] > x[right + 1])
                if left_ok and right_ok:
                    center = left + 1 + length // 2
                    peaks.append(center)
                    i = right  # skip the entire plateau
                    continue
        i += 1

    # ---- Step 2: prominence and width filters ----
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

            # Left base – full search to the bound
            left_min = peak_val
            for i in range(p - 1, left_bound - 1, -1):
                if x[i] < left_min:
                    left_min = x[i]

            # Right base
            right_min = peak_val
            for i in range(p + 1, right_bound + 1):
                if x[i] < right_min:
                    right_min = x[i]

            prom = peak_val - max(left_min, right_min)
            if prom < prominence:
                continue

            # Width at rel_height
            if width >= 0:
                h = peak_val - prom * rel_height
                # Find left intersection (closest to p)
                wl = p
                for i in range(p, left_bound - 1, -1):
                    if x[i] <= h:
                        wl = i
                        break
                # Find right intersection (closest to p)
                wr = p
                for i in range(p, right_bound + 1):
                    if x[i] <= h:
                        wr = i
                        break
                w = wr - wl
                if w < width:
                    continue

            filtered.append(p)
        peaks = filtered

    # ---- Step 3: distance filter (keep highest peak among those too close) ----
    if distance > 1 and len(peaks) > 1:
        # Build groups of peaks that violate the distance constraint
        # We'll walk through the sorted list and whenever the next peak is too close,
        # we add it to a group and later keep the one with the highest value.
        new_peaks = List.empty_list(int64)
        i = 0
        while i < len(peaks):
            group = List.empty_list(int64)
            group.append(peaks[i])
            j = i + 1
            while j < len(peaks) and peaks[j] - peaks[i] < distance:
                group.append(peaks[j])
                j += 1
            # Now select the peak with the highest value from the group
            best_idx = group[0]
            best_val = x[best_idx]
            for k in range(1, len(group)):
                if x[group[k]] > best_val:
                    best_idx = group[k]
                    best_val = x[best_idx]
            new_peaks.append(best_idx)
            i = j  # move to next group
        peaks = new_peaks

    # Convert to numpy array and sort (just in case)
    out = np.empty(len(peaks), dtype=np.int64)
    for i in range(len(peaks)):
        out[i] = peaks[i]
    out.sort()
    return out


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
    """
    Detect peaks (in high) and valleys (in low) using custom Numba peak detection.

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
    # Input validation: check for NaNs
    if np.any(np.isnan(high)):
        raise ValueError("high array contains NaNs")
    if np.any(np.isnan(low)):
        raise ValueError("low array contains NaNs")

    # Convert optional parameters to sentinel values expected by Numba
    plateau = plateau_size if plateau_size is not None else -1
    width_ = width if width is not None else -1.0
    wlen_ = wlen if wlen is not None else -1

    # Ensure arrays are float64 and contiguous
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)

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
    """
    Universal Zigzag indicator (returns numpy arrays of indices).
    """
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
    high_col: str = "high",
    low_col: str = "low",
    date_col: str = "date",
    prominence_peak: float = 0.01,
    prominence_valley: float = 0.01,
    distance: int = 5,
    width: float | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add boolean columns 'is_peak' and 'is_valley' to the Polars DataFrame.

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
        Original DataFrame with two new columns: 'is_peak{suffix}', 'is_valley{suffix}'.
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

    suffix = suffix or ""
    return pl.DataFrame({
        date_col: df[date_col],
        f"is_peak{suffix}": is_peak,
        f"is_valley{suffix}": is_valley,
    })