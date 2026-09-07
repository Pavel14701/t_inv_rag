# -*- coding: utf-8 -*-
"""Unit tests for the online (repaint-free) ZigZag.

Tests cover:
- parity: online confirmed pivots == historical batch ZigZag on
  synthetic data (random walks, trends, waves, plateaus)
- prefix consistency: feeding data bar-by-bar equals feeding prefixes,
  i.e. confirmed pivots never change (no repaint)
- confirmation semantics: pivot confirmed only after the price reversed
  by at least the threshold; confirm_idx > pivot_idx
- relative (pct) reversal thresholds
- edge cases: flat series, single bar, validation errors
- relation to the scipy-style historical ZigZag on clean swings
"""
import numpy as np
import pytest

from ...custom.market_structure import (
    OnlineZigZag,
    Pivot,
    zigzag_reversal_numpy,
)

REVERSAL = 2.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _sine_ohlc(n: int = 300, amplitude: float = 10.0, period: float = 40.0):
    """Clean sine swings: high = low + small constant spread."""
    t = np.arange(n, dtype=np.float64)
    mid = 100.0 + amplitude * np.sin(2 * np.pi * t / period)
    return mid + 0.1, mid - 0.1  # high, low


def _random_walk_ohlc(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    spread = np.abs(rng.normal(0.5, 0.2, n))
    return close + spread, close - spread


def _pivots_to_arrays(pivots: list[Pivot]):
    kinds = np.array([p.kind for p in pivots], dtype=int)
    idxs = np.array([p.idx for p in pivots], dtype=int)
    peaks = idxs[kinds == 1]
    valleys = idxs[kinds == -1]
    return peaks, valleys


# -----------------------------------------------------------------------------
# Parity: online == historical batch
# -----------------------------------------------------------------------------
@pytest.mark.custom
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_online_matches_historical_random_walk(seed: int) -> None:
    high, low = _random_walk_ohlc(500, seed=seed)
    online = OnlineZigZag(REVERSAL)
    online.update_series(high, low)
    historical = zigzag_reversal_numpy(high, low, REVERSAL)
    assert online.confirmed == historical
    assert len(historical) > 5


@pytest.mark.custom
def test_online_matches_historical_clean_waves() -> None:
    high, low = _sine_ohlc()
    online = OnlineZigZag(REVERSAL)
    online.update_series(high, low)
    historical = zigzag_reversal_numpy(high, low, REVERSAL)
    assert online.confirmed == historical
    peaks, valleys = _pivots_to_arrays(historical)
    # alternating peaks/valleys
    kinds = [p.kind for p in historical]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert len(peaks) >= 3 and len(valleys) >= 3


@pytest.mark.custom
def test_online_matches_historical_trending() -> None:
    """In a steady up-trend only the initial valley is ever confirmed:
    the last (open) leg extreme stays pending until a real reversal.
    """
    n = 100
    close = 100.0 + np.arange(n, dtype=np.float64)
    high, low = close + 0.1, close - 0.1
    online = OnlineZigZag(REVERSAL)
    online.update_series(high, low)
    assert online.confirmed == zigzag_reversal_numpy(high, low, REVERSAL)
    assert len(online.confirmed) == 1
    v = online.confirmed[0]
    assert v.kind == -1 and v.idx == 0
    assert v.confirm_idx >= 2  # first bar reaching the reversal


# -----------------------------------------------------------------------------
# No-repaint: confirmed prefix is immutable
# -----------------------------------------------------------------------------
@pytest.mark.custom
def test_confirmed_pivots_never_repaint() -> None:
    high, low = _random_walk_ohlc(400, seed=5)
    full = OnlineZigZag(REVERSAL)
    full.update_series(high, low)
    final = full.confirmed
    # run prefixes with fresh instances: every confirmed pivot of a
    # prefix must equal the corresponding pivot of the full run
    for t in (50, 100, 200, 300, 399):
        part = OnlineZigZag(REVERSAL)
        part.update_series(high[:t + 1], low[:t + 1])
        got = part.confirmed
        assert len(got) <= len(final)
        assert got == final[:len(got)]


@pytest.mark.custom
def test_incremental_equals_batch_prefix() -> None:
    high, low = _random_walk_ohlc(250, seed=9)
    zz = OnlineZigZag(REVERSAL)
    for t in range(len(high)):
        zz.update(high[t], low[t])
        historical = zigzag_reversal_numpy(high[:t + 1], low[:t + 1], REVERSAL)
        assert zz.confirmed == historical


# -----------------------------------------------------------------------------
# Confirmation semantics
# -----------------------------------------------------------------------------
@pytest.mark.custom
def test_pivot_confirmed_only_after_reversal() -> None:
    # explicit zig-zag: 100 -> 110 (peak) -> down; the peak is confirmed
    # only once price dropped >= 2 from 110.5
    highs = [100.5] * 5 + [110.5] * 3 + [109.0, 107.5]
    lows = [99.5] * 5 + [109.5] * 3 + [107.9, 106.5]
    zz = OnlineZigZag(2.0)
    emitted: dict[int, list[Pivot]] = {}
    for i, (h, l) in enumerate(zip(highs, lows)):
        pivots = zz.update(h, l)
        if pivots:
            emitted[i] = pivots
    # bar 5: the init-phase valley of the flat start is anchored on the
    # first strong up-move (110.5 - 99.5 = 11 >= 2)
    # bar 8: the peak is emitted exactly on the reversal bar:
    # 110.5 - 107.9 = 2.6 >= 2
    assert set(emitted) == {5, 8}
    assert emitted[5] == [Pivot(0, 99.5, -1, 5)]
    assert emitted[8] == [Pivot(5, 110.5, +1, 8)]  # first bar of the plateau
    assert zz.confirmed == [
        Pivot(0, 99.5, -1, 5),
        Pivot(5, 110.5, +1, 8),
    ]


@pytest.mark.custom
def test_confirm_idx_greater_than_pivot_idx() -> None:
    high, low = _random_walk_ohlc(300, seed=11)
    zz = OnlineZigZag(REVERSAL)
    zz.update_series(high, low)
    for p in zz.confirmed:
        assert p.confirm_idx >= p.idx


@pytest.mark.custom
def test_relative_reversal_threshold() -> None:
    # leg extreme 200.2; 1% pct threshold -> reversal must exceed ~2
    highs = [100.2] * 3 + [200.2] * 4 + [198.5]
    lows = [99.8] * 3 + [199.8] * 4 + [198.4]
    zz = OnlineZigZag(1.0, reversal_pct=0.01)
    for h, l in zip(highs, lows):
        zz.update(h, l)
    peaks = [p for p in zz.confirmed if p.kind == +1]
    # drop of 200.2 - 198.4 = 1.8: below pct threshold
    # max(1.0, 0.01 * 200.2) = 2.002 -> no peak yet
    # (a pure absolute threshold of 1.0 WOULD have confirmed it)
    assert peaks == []
    highs.append(198.5)
    lows.append(196.5)  # drop 200.2 - 196.5 = 3.7 >= 2.002 -> confirmed
    zz = OnlineZigZag(1.0, reversal_pct=0.01)
    for h, l in zip(highs, lows):
        zz.update(h, l)
    peaks = [p for p in zz.confirmed if p.kind == +1]
    assert peaks and peaks[-1].price == 200.2


# -----------------------------------------------------------------------------
# Relation to the scipy-style historical ZigZag
# -----------------------------------------------------------------------------
@pytest.mark.custom
def test_matches_scipy_zigzag_on_clean_swings() -> None:
    """On clean sine swings the online pivots match the historical
    scipy-style ZigZag, minus the final (still open) leg extremes.
    """
    from ...trend.zigzag import zigzag_peaks_valleys

    high, low = _sine_ohlc(400, amplitude=15.0, period=50.0)
    zz = OnlineZigZag(1.0)
    zz.update_series(high, low)
    peaks, valleys = _pivots_to_arrays(zz.confirmed)
    ref_peaks, ref_valleys = zigzag_peaks_valleys(
        high, low,
        prominence_peak=0.0, prominence_valley=0.0, distance=1,
        width=None, wlen=None, rel_height=0.5, plateau_size=None,
    )
    # every confirmed online pivot must exist in the historical result.
    # The online machine also confirms the initial series extreme (the
    # bar-0 valley) during its init phase; the scipy-style ZigZag only
    # reports alternating pivots and has no such anchor, so skip idx 0.
    for p in peaks:
        assert p in ref_peaks
    for v in valleys:
        if v == 0:
            continue
        assert v in ref_valleys
    # and the online result must lag by at most one peak/valley per kind
    assert len(peaks) >= len(ref_peaks) - 2
    assert len(valleys) >= len(ref_valleys) - 2


# -----------------------------------------------------------------------------
# Edge cases and validation
# -----------------------------------------------------------------------------
@pytest.mark.custom
def test_flat_series_no_pivots() -> None:
    high = np.full(50, 100.5)
    low = np.full(50, 99.5)
    zz = OnlineZigZag(2.0)
    zz.update_series(high, low)
    assert zz.confirmed == []
    assert zz.pending is None


@pytest.mark.custom
def test_single_bar() -> None:
    # range 2 < threshold 5 -> nothing confirmed
    zz = OnlineZigZag(5.0)
    assert zz.update(101.0, 99.0) == []
    assert zz.confirmed == []
    # range >= threshold: the bar close already carries the info, so a
    # same-bar confirmation is legitimate (confirm_idx == pivot idx)
    zz2 = OnlineZigZag(1.0)
    new = zz2.update(101.0, 99.0)
    assert len(new) == 1 and new[0].confirm_idx == new[0].idx == 0


@pytest.mark.custom
def test_invalid_reversal_raises() -> None:
    with pytest.raises(ValueError, match='reversal'):
        OnlineZigZag(0.0)
    with pytest.raises(ValueError, match='reversal'):
        OnlineZigZag(-1.0)
    with pytest.raises(ValueError, match='reversal_pct'):
        OnlineZigZag(1.0, reversal_pct=1.5)


@pytest.mark.custom
def test_non_finite_input_raises() -> None:
    zz = OnlineZigZag(1.0)
    zz.update(100.0, 99.0)
    with pytest.raises(ValueError, match='finite'):
        zz.update(np.nan, 99.0)
    with pytest.raises(ValueError, match='finite'):
        zz.update(101.0, np.inf)


@pytest.mark.custom
def test_update_series_length_mismatch() -> None:
    zz = OnlineZigZag(1.0)
    with pytest.raises(ValueError, match='same length'):
        zz.update_series(np.ones(3), np.ones(4))
