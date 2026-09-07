# -*- coding: utf-8 -*-
"""Unit tests for the ZigZag module (peak/valley detection).

Tests cover:
- peak detection against scipy.signal.find_peaks (exact parity on a
  rounded random walk, which naturally contains flat tops)
- plateau detection regressions (flat tops, short plateaus, edge cases)
- distance filter regression (scipy-style greedy)
- valleys via inverted low, prominence filter
- zigzag_numpy validation and edge cases
- zigzag_ind (Polars Series input), zigzag_peaks_valleys sentinels
- zigzag_polars (added columns, no mutation, suffix, custom columns)
"""

import numpy as np
import polars as pl
import pytest
from scipy.signal import find_peaks

from ...trend.zigzag import (
    zigzag_ind,
    zigzag_numpy,
    zigzag_polars,
    zigzag_peaks_valleys,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _peaks_of(x, **kwargs) -> np.ndarray:
    """Peaks of a single series via zigzag_peaks_valleys."""
    x = np.asarray(x, dtype=np.float64)
    peaks, _ = zigzag_peaks_valleys(
        x, x,
        prominence_peak=kwargs.get('prom', 0.0),
        prominence_valley=0.0,
        distance=kwargs.get('dist', 1),
        width=kwargs.get('width'),
        wlen=kwargs.get('wlen'),
        rel_height=kwargs.get('rel_height', 0.5),
        plateau_size=kwargs.get('ps'),
    )
    return peaks


def _valleys_of(x, **kwargs) -> np.ndarray:
    """Valleys of a single series via zigzag_peaks_valleys."""
    x = np.asarray(x, dtype=np.float64)
    _, valleys = zigzag_peaks_valleys(
        x, x,
        prominence_peak=0.0,
        prominence_valley=kwargs.get('prom', 0.0),
        distance=kwargs.get('dist', 1),
        width=kwargs.get('width'),
        wlen=kwargs.get('wlen'),
        rel_height=kwargs.get('rel_height', 0.5),
        plateau_size=kwargs.get('ps'),
    )
    return valleys


@pytest.fixture
def walk() -> np.ndarray:
    """Rounded random walk: flat tops occur naturally."""
    rng = np.random.default_rng(0)
    return np.round(np.cumsum(rng.normal(0, 1, 500)), 1)


# -----------------------------------------------------------------------------
# scipy parity
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_peaks_scipy_parity_no_plateau_size(walk: np.ndarray) -> None:
    """Peaks match scipy exactly (default plateau handling)."""
    ours = _peaks_of(walk)
    ref = find_peaks(walk)[0]
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
@pytest.mark.parametrize('ps', [1, 2, 3, 5])
def test_zigzag_peaks_scipy_parity_plateau_size(
    walk: np.ndarray, ps: int,
) -> None:
    """plateau_size semantics match scipy (midpoint, min run length)."""
    ours = _peaks_of(walk, ps=ps)
    ref = find_peaks(walk, plateau_size=ps)[0]
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
def test_zigzag_valleys_scipy_parity(walk: np.ndarray) -> None:
    """Valleys (inverted low) match scipy exactly."""
    ours = _valleys_of(walk)
    ref = find_peaks(-walk)[0]
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
def test_zigzag_prominence_scipy_parity(walk: np.ndarray) -> None:
    """Prominence filter matches scipy (full-window prominence)."""
    ours = _peaks_of(walk, prom=2.0)
    ref = find_peaks(walk, prominence=2.0)[0]
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
def test_zigzag_distance_scipy_parity(walk: np.ndarray) -> None:
    """Distance filter matches scipy on a natural series."""
    ours = _peaks_of(walk, dist=10)
    ref = find_peaks(walk, distance=10)[0]
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
@pytest.mark.parametrize('dist', [2, 3, 7, 15, 40])
def test_zigzag_distance_property(walk: np.ndarray, dist: int) -> None:
    """All surviving peaks are at least `dist` bars apart."""
    peaks = _peaks_of(walk, dist=dist)
    assert len(peaks) > 0
    assert (np.diff(peaks) >= dist).all()


# -----------------------------------------------------------------------------
# Plateau regression tests (bugs fixed in _find_peaks_nb)
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_plateau_len2_detected_without_plateau_size() -> None:
    """Regression: a 2-bar flat top was never detected before."""
    x = np.array([1.0, 3.0, 3.0, 2.0, 1.0])
    np.testing.assert_array_equal(_peaks_of(x), [1])


@pytest.mark.trend
def test_zigzag_plateau_len3_detected_without_plateau_size() -> None:
    """Regression: a 3-bar flat top was never detected before."""
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    np.testing.assert_array_equal(_peaks_of(x), [3])


@pytest.mark.trend
def test_zigzag_plateau_next_to_higher_bar_is_peak() -> None:
    """Regression: plateau was missed because of a higher bar further away.

    [5, 1, 2, 2, 2, 1, 0]: the plateau 2..2 IS a local maximum
    (immediate neighbours are lower), the distant 5 is irrelevant.
    """
    x = np.array([5.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    np.testing.assert_array_equal(_peaks_of(x, ps=1), [3])


@pytest.mark.trend
@pytest.mark.parametrize('x', [
    [3.0, 2.0, 2.0, 2.0, 1.0],   # higher bar on the left
    [0.0, 2.0, 2.0, 2.0, 3.0],   # higher bar on the right
])
def test_zigzag_plateau_below_neighbour_is_not_peak(x: list) -> None:
    """Regression: plateau lower than an adjacent bar was a false positive."""
    np.testing.assert_array_equal(_peaks_of(np.array(x), ps=1), [])


@pytest.mark.trend
@pytest.mark.parametrize('x', [
    [2.0, 2.0, 2.0, 1.0, 0.0],   # plateau touches the start
    [0.0, 1.0, 2.0, 2.0, 2.0],   # plateau touches the end
])
def test_zigzag_plateau_at_edge_is_not_peak(x: list) -> None:
    """Regression: edge plateau was a false positive (no lower neighbour)."""
    np.testing.assert_array_equal(_peaks_of(np.array(x), ps=1), [])


@pytest.mark.trend
def test_zigzag_plateau_midpoint_is_scipy_convention() -> None:
    """Peak of an even-length plateau is the scipy midpoint (left+right)//2."""
    x = np.array([0.0, 1.0, 2.0, 2.0, 1.0])  # plateau at 2..3
    np.testing.assert_array_equal(_peaks_of(x), [2])  # (2+3)//2


# -----------------------------------------------------------------------------
# Simple hand-crafted cases
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_simple_alternating_series() -> None:
    """Alternating highs/lows give alternating peaks and valleys."""
    high = np.array([1.0, 3.0, 1.0, 4.0, 1.0, 2.0, 1.0])
    low = np.array([1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0])
    peaks, valleys = zigzag_numpy(high, low, prominence_peak=0.0,
                                  prominence_valley=0.0, distance=1)
    np.testing.assert_array_equal(peaks, [1, 3, 5])
    np.testing.assert_array_equal(valleys, [2, 4])


@pytest.mark.trend
def test_zigzag_monotonic_series_has_no_extremes() -> None:
    """Strictly increasing / decreasing series: no peaks, no valleys."""
    x = np.arange(20, dtype=np.float64)
    peaks, valleys = zigzag_numpy(x, x, prominence_peak=0.0,
                                  prominence_valley=0.0, distance=1)
    assert peaks.size == 0
    assert valleys.size == 0


@pytest.mark.trend
def test_zigzag_constant_series_has_no_extremes() -> None:
    """All-equal series: no peaks, no valleys (flat is not a maximum)."""
    x = np.full(20, 5.0)
    peaks, valleys = zigzag_numpy(x, x, prominence_peak=0.0,
                                  prominence_valley=0.0, distance=1)
    assert peaks.size == 0
    assert valleys.size == 0


@pytest.mark.trend
def test_zigzag_strict_ieee_matches_scipy(walk: np.ndarray) -> None:
    """The kernel is compiled without fastmath (strict IEEE 754).

    Guard against a future re-enablement: results must stay identical
    to scipy, which uses strict IEEE semantics.  (width is excluded:
    our simplified width metric is not scipy-exact by design and is
    covered by property tests instead.)
    """
    for kwargs in ({}, {'prom': 2.0}, {'dist': 10}, {'ps': 1},
                   {'prom': 1.0, 'dist': 5}):
        ours = _peaks_of(walk, **kwargs)
        ref_kwargs = {
            'prominence': kwargs.get('prom'),
            'distance': kwargs.get('dist'),
            'plateau_size': kwargs.get('ps'),
        }
        ref_kwargs = {k: v for k, v in ref_kwargs.items() if v is not None}
        ref = find_peaks(walk, **ref_kwargs)[0]
        np.testing.assert_array_equal(ours, ref)


@pytest.mark.trend
def test_zigzag_endpoints_are_never_extremes() -> None:
    """First and last bars can never be peaks or valleys."""
    rng = np.random.default_rng(7)
    x = np.round(np.cumsum(rng.normal(0, 1, 300)), 1)
    for arr in (_peaks_of(x), _valleys_of(x)):
        assert arr.size == 0 or (arr[0] >= 1 and arr[-1] <= len(x) - 2)


# -----------------------------------------------------------------------------
# Distance filter regression (greedy)
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_distance_greedy_removes_cross_group_peaks() -> None:
    """Regression: peaks within `distance` of the KEPT peak were kept.

    Peaks at 1 (val 5), 4 (val 6), 7 (val 5.5), distance=5:
    the highest peak 4 must remove both 1 and 7 -> only [4] survives.
    """
    x = np.array([0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.5, 0.0])
    np.testing.assert_array_equal(_peaks_of(x, dist=5), [4])


@pytest.mark.trend
def test_zigzag_distance_keeps_highest_in_chain() -> None:
    """A chain of close peaks collapses to its highest member."""
    x = np.array([0.0, 1.0, 0.0, 9.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(_peaks_of(x, dist=4), [3])


@pytest.mark.trend
def test_zigzag_distance_one_means_no_filter() -> None:
    """distance=1 (and default None handling) keeps all peaks."""
    x = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    np.testing.assert_array_equal(_peaks_of(x, dist=1), [1, 3, 5])


# -----------------------------------------------------------------------------
# Prominence / width filters
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_prominence_filters_small_peaks() -> None:
    """Only peaks with enough prominence survive."""
    x = np.array([0.0, 5.0, 0.0, 6.0, 0.0])  # proms: 5, 6
    assert np.isin(_peaks_of(x, prom=5.5), [3]).all()
    np.testing.assert_array_equal(_peaks_of(x, prom=1.0), [1, 3])


@pytest.mark.trend
def test_zigzag_width_filters_narrow_peaks() -> None:
    """width=0 (disabled vs enabled sentinel) and huge width."""
    x = np.array([0.0, 3.0, 0.0, 4.0, 0.0, 3.0, 0.0])
    # Wide peaks in a triangle wave: generous width keeps them...
    kept = _peaks_of(x, width=1.0)
    np.testing.assert_array_equal(kept, [1, 3, 5])
    # ...but an impossible width drops everything.
    assert _peaks_of(x, width=100.0).size == 0


@pytest.mark.trend
def test_zigzag_wlen_limits_prominence_window() -> None:
    """Small wlen lowers measured prominence and can filter the peak.

    Single mountain, peak 10 at index 5: full-window prominence is
    10 - 4 = 6, but with wlen=5 the window is [3..7] and prominence
    drops to 10 - 6 = 4.
    """
    x = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 8.5, 7.0, 5.5, 4.0])
    assert _peaks_of(x, prom=5.0).size == 1     # full window: survives
    assert _peaks_of(x, prom=5.0, wlen=5).size == 0  # narrow: filtered out


# -----------------------------------------------------------------------------
# zigzag_numpy: validation and edge cases
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_numpy_rejects_nan() -> None:
    """NaN in high or low raises ValueError."""
    x = np.arange(10.0)
    x_nan = x.copy()
    x_nan[3] = np.nan
    with pytest.raises(ValueError, match='high.*NaN'):
        zigzag_numpy(x_nan, x, prominence_peak=0.0, prominence_valley=0.0,
                     distance=1)
    with pytest.raises(ValueError, match='low.*NaN'):
        zigzag_numpy(x, x_nan, prominence_peak=0.0, prominence_valley=0.0,
                     distance=1)


@pytest.mark.trend
def test_zigzag_numpy_rejects_inf() -> None:
    """Inf in high or low raises ValueError (never reaches the kernel)."""
    x = np.arange(10.0)
    x_inf = x.copy()
    x_inf[3] = np.inf
    with pytest.raises(ValueError, match='Inf'):
        zigzag_numpy(x_inf, x, prominence_peak=0.0, prominence_valley=0.0,
                     distance=1)
    with pytest.raises(ValueError, match='Inf'):
        zigzag_numpy(x, x_inf, prominence_peak=0.0, prominence_valley=0.0,
                     distance=1)


@pytest.mark.trend
def test_zigzag_numpy_rejects_length_mismatch() -> None:
    """Regression: different lengths of high/low were not validated."""
    x = np.arange(10.0)
    with pytest.raises(ValueError, match='same length'):
        zigzag_numpy(x, x[:5], prominence_peak=0.0, prominence_valley=0.0,
                     distance=1)


@pytest.mark.trend
def test_zigzag_numpy_short_series_returns_empty() -> None:
    """Series shorter than 3 bars gracefully return empty index arrays."""
    for n in (0, 1, 2):
        peaks, valleys = zigzag_numpy(
            np.ones(n), np.ones(n),
            prominence_peak=0.0, prominence_valley=0.0, distance=1,
        )
        assert peaks.size == 0 and valleys.size == 0


@pytest.mark.trend
def test_zigzag_numpy_result_dtypes_and_sorted() -> None:
    """Results are int64 arrays sorted ascending."""
    rng = np.random.default_rng(3)
    x = np.round(np.cumsum(rng.normal(0, 1, 200)), 1)
    peaks, valleys = zigzag_numpy(x, x, prominence_peak=0.0,
                                  prominence_valley=0.0, distance=1)
    assert peaks.dtype == np.int64 and valleys.dtype == np.int64
    assert (np.diff(peaks) > 0).all()
    assert (np.diff(valleys) > 0).all()


@pytest.mark.trend
def test_zigzag_numpy_accepts_non_contiguous_input() -> None:
    """Non-contiguous (strided) arrays are handled via ascontiguousarray."""
    x = np.arange(40.0)[::2]  # stride 2, non-contiguous
    x[10] = 100.0  # guaranteed interior peak
    peaks, valleys = zigzag_numpy(x, x, prominence_peak=0.0,
                                  prominence_valley=0.0, distance=1)
    assert 10 in peaks
    assert peaks.size >= 1


# -----------------------------------------------------------------------------
# Wrappers: zigzag_peaks_valleys / zigzag_ind
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_zigzag_peaks_valleys_none_sentinels() -> None:
    """None width/wlen/plateau_size behave like the disabled sentinels."""
    rng = np.random.default_rng(11)
    x = np.round(np.cumsum(rng.normal(0, 1, 200)), 1)
    via_none = zigzag_peaks_valleys(x, x, 0.0, 0.0, 1, None, None, 0.5, None)
    via_sentinel = zigzag_peaks_valleys(x, x, 0.0, 0.0, 1, -1.0, -1, 0.5, -1)
    np.testing.assert_array_equal(via_none[0], via_sentinel[0])
    np.testing.assert_array_equal(via_none[1], via_sentinel[1])


@pytest.mark.trend
def test_zigzag_ind_matches_numpy(walk: np.ndarray) -> None:
    """zigzag_ind on numpy input matches zigzag_numpy."""
    ind_res = zigzag_ind(walk, walk, prominence_peak=0.0,
                         prominence_valley=0.0, distance=1)
    numpy_res = zigzag_numpy(walk, walk, prominence_peak=0.0,
                             prominence_valley=0.0, distance=1)
    np.testing.assert_array_equal(ind_res[0], numpy_res[0])
    np.testing.assert_array_equal(ind_res[1], numpy_res[1])


@pytest.mark.trend
def test_zigzag_ind_accepts_polars_series(walk: np.ndarray) -> None:
    """zigzag_ind accepts pl.Series and matches numpy input."""
    series_res = zigzag_ind(pl.Series(walk), pl.Series(walk),
                            prominence_peak=0.0, prominence_valley=0.0,
                            distance=1)
    numpy_res = zigzag_numpy(walk, walk, prominence_peak=0.0,
                             prominence_valley=0.0, distance=1)
    np.testing.assert_array_equal(series_res[0], numpy_res[0])
    np.testing.assert_array_equal(series_res[1], numpy_res[1])


@pytest.mark.trend
def test_zigzag_ind_default_parameters_run(walk: np.ndarray) -> None:
    """Defaults (prominence=0.01, distance=5) execute and return arrays."""
    peaks, valleys = zigzag_ind(walk, walk)
    assert isinstance(peaks, np.ndarray)
    assert isinstance(valleys, np.ndarray)
    if peaks.size > 1:
        assert (np.diff(peaks) >= 5).all()


# -----------------------------------------------------------------------------
# zigzag_polars
# -----------------------------------------------------------------------------
@pytest.fixture
def df_ohlc_small() -> pl.DataFrame:
    rng = np.random.default_rng(5)
    close = np.round(np.cumsum(rng.normal(0, 1, 120)), 1)
    high = close + np.abs(rng.normal(0, 0.5, 120))
    low = close - np.abs(rng.normal(0, 0.5, 120))
    return pl.DataFrame({'high': high, 'low': low, 'close': close})


@pytest.mark.trend
def test_zigzag_polars_adds_boolean_columns(df_ohlc_small: pl.DataFrame) -> None:
    """is_peak/is_valley columns are added, boolean, full length."""
    result = zigzag_polars(df_ohlc_small, prominence_peak=0.0,
                           prominence_valley=0.0, distance=1)
    assert 'is_peak' in result.columns
    assert 'is_valley' in result.columns
    assert result.height == df_ohlc_small.height
    assert result['is_peak'].dtype == pl.Boolean
    assert result['is_valley'].dtype == pl.Boolean
    assert result['is_peak'].sum() > 0
    assert result['is_valley'].sum() > 0


@pytest.mark.trend
def test_zigzag_polars_matches_numpy(df_ohlc_small: pl.DataFrame) -> None:
    """Boolean masks mark exactly the indices returned by zigzag_numpy."""
    result = zigzag_polars(df_ohlc_small, prominence_peak=0.0,
                           prominence_valley=0.0, distance=1)
    peak_idx, valley_idx = zigzag_numpy(
        df_ohlc_small['high'].to_numpy(),
        df_ohlc_small['low'].to_numpy(),
        prominence_peak=0.0, prominence_valley=0.0, distance=1,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(result['is_peak'].to_numpy()), peak_idx)
    np.testing.assert_array_equal(
        np.flatnonzero(result['is_valley'].to_numpy()), valley_idx)


@pytest.mark.trend
def test_zigzag_polars_does_not_mutate_input(
    df_ohlc_small: pl.DataFrame,
) -> None:
    """The input DataFrame is not modified in place."""
    before = df_ohlc_small.columns
    zigzag_polars(df_ohlc_small)
    assert df_ohlc_small.columns == before
    assert 'is_peak' not in df_ohlc_small.columns


@pytest.mark.trend
def test_zigzag_polars_suffix(df_ohlc_small: pl.DataFrame) -> None:
    """suffix is appended to the new column names."""
    result = zigzag_polars(df_ohlc_small, suffix='_zz')
    assert 'is_peak_zz' in result.columns
    assert 'is_valley_zz' in result.columns
    assert 'is_peak' not in result.columns


@pytest.mark.trend
def test_zigzag_polars_custom_column_names(df_ohlc_small: pl.DataFrame) -> None:
    """Custom high/low column names are honoured."""
    renamed = df_ohlc_small.rename({'high': 'h', 'low': 'l'})
    result = zigzag_polars(
        renamed, high_col='h', low_col='l',
        prominence_peak=0.0, prominence_valley=0.0, distance=1,
    )
    assert 'is_peak' in result.columns