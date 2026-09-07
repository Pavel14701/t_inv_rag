# -*- coding: utf-8 -*-
"""Online (incremental, repaint-free) ZigZag.

The historical ZigZag recomputes pivots over the whole series, so the
latest pivot can move as new bars arrive (*repainting*).  This module
provides :class:`OnlineZigZag`, an incremental state machine that only
*confirms* a pivot once price has reversed from it by at least the
reversal threshold.  A confirmed pivot can never change afterwards,
which makes the output safe for live trading and backtests without
look-ahead bias.

Semantics
---------
- A pivot is confirmed at bar ``c >= pivot_idx`` (equal only when a
  single bar spans the whole reversal): from bar ``c`` onwards
  the pivot is final (``confirm_idx = c``).
- ``zigzag_reversal_numpy`` applies the identical rules in one batch
  pass and is used as the historical reference for parity tests.
- The pending (latest, unconfirmed) leg extreme is NOT returned as a
  pivot - it may still move or disappear.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pivot:
    """A confirmed ZigZag pivot."""

    idx: int            # bar of the extreme
    price: float        # extreme price
    kind: int           # +1 peak (in high), -1 valley (in low)
    confirm_idx: int    # bar at which the pivot became final


@dataclass
class _Leg:
    """Tentative extreme of the current leg (may still move)."""

    idx: int
    value: float


class OnlineZigZag:
    """Incremental ZigZag with confirmed (repaint-free) pivots.

    Parameters
    ----------
    reversal : float
        Minimum absolute price reversal that finalises a pivot.
    reversal_pct : float, optional
        If given, the threshold for a leg starting at ``value`` becomes
        ``max(reversal, reversal_pct * value)`` (relative component).

    """

    def __init__(
        self,
        reversal: float = 0.01,
        reversal_pct: float | None = None,
    ) -> None:
        if reversal <= 0:
            raise ValueError(f'reversal must be positive; got {reversal}')
        if reversal_pct is not None and not 0 < reversal_pct < 1:
            raise ValueError(
                f'reversal_pct must be in (0, 1); got {reversal_pct}'
            )
        self.reversal = float(reversal)
        self.reversal_pct = reversal_pct
        self._confirmed: list[Pivot] = []
        self._direction = 0          # 0 undetermined, +1 up leg, -1 down leg
        self._leg = _Leg(-1, np.nan)  # tentative extreme of the current leg
        self._other = _Leg(-1, np.nan)  # opposite extreme (init phase)
        self._n = 0                  # bars processed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def confirmed(self) -> list[Pivot]:
        """Confirmed pivots so far (immutable prefix, never repaints)."""
        return list(self._confirmed)

    @property
    def pending(self) -> _Leg | None:
        """Tentative extreme of the current leg (may still move)."""
        if self._direction == 0 or self._leg.idx < 0:
            return None
        return self._leg

    def update(self, high: float, low: float) -> list[Pivot]:
        """Feed the next bar; return pivots confirmed by this bar."""
        if not (np.isfinite(high) and np.isfinite(low)):
            raise ValueError('high/low must be finite numbers')
        i = self._n
        self._n += 1
        new_pivots: list[Pivot] = []
        thr_h = self._threshold(high)
        thr_l = self._threshold(low)

        if self._direction == 0:
            # Init: track both extremes until one triggers a reversal.
            if not np.isfinite(self._leg.value) or high > self._leg.value:
                self._leg = _Leg(i, high)
            if not np.isfinite(self._other.value) or low < self._other.value:
                self._other = _Leg(i, low)
            if self._leg.value - low >= thr_h:
                # Reversal down from the running high: peak confirmed.
                new_pivots.append(
                    Pivot(self._leg.idx, self._leg.value, +1, i)
                )
                self._direction = -1
                self._leg = _Leg(i, low)
                self._other = _Leg(-1, np.nan)
            elif high - self._other.value >= thr_l:
                new_pivots.append(
                    Pivot(self._other.idx, self._other.value, -1, i)
                )
                self._direction = +1
                self._leg = _Leg(i, high)
                self._other = _Leg(-1, np.nan)
        elif self._direction == 1:
            if high > self._leg.value:
                self._leg = _Leg(i, high)
            elif self._leg.value - low >= self._threshold(self._leg.value):
                new_pivots.append(
                    Pivot(self._leg.idx, self._leg.value, +1, i)
                )
                self._direction = -1
                self._leg = _Leg(i, low)
        else:
            if low < self._leg.value:
                self._leg = _Leg(i, low)
            elif high - self._leg.value >= self._threshold(self._leg.value):
                new_pivots.append(
                    Pivot(self._leg.idx, self._leg.value, -1, i)
                )
                self._direction = +1
                self._leg = _Leg(i, high)
        self._confirmed.extend(new_pivots)
        return new_pivots

    def update_series(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> list[Pivot]:
        """Feed a whole series; return all newly confirmed pivots."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        if len(high) != len(low):
            raise ValueError('high and low must have the same length')
        out: list[Pivot] = []
        for h, lo in zip(high, low):
            out.extend(self.update(float(h), float(lo)))
        return out

    # ------------------------------------------------------------------
    def _threshold(self, value: float) -> float:
        if self.reversal_pct is None:
            return self.reversal
        return max(self.reversal, self.reversal_pct * abs(value))


# ----------------------------------------------------------------------
# Batch (historical) reference implementation
# ----------------------------------------------------------------------
def zigzag_reversal_numpy(
    high: np.ndarray,
    low: np.ndarray,
    reversal: float,
    reversal_pct: float | None = None,
) -> list[Pivot]:
    """Offline reversal ZigZag - the historical twin of :class:`OnlineZigZag`.

    Applies the same confirmation rules in a single batch pass over the
    full arrays.  By construction the result equals the confirmed pivots
    of :class:`OnlineZigZag` fed with the same data (see tests).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if len(high) != len(low):
        raise ValueError('high and low must have the same length')
    zz = OnlineZigZag(reversal, reversal_pct)
    for h, lo in zip(high, low):
        zz.update(float(h), float(lo))
    return zz.confirmed


def confirmed_pivot_arrays(
    pivots: list[Pivot],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split pivots into (peaks, valleys, confirm_idx, next_extreme_idx).

    ``next_extreme_idx`` is aligned with the merged pivot order: for
    each pivot it holds the bar of the next confirmed extreme (either
    kind), or sentinel ``-1`` when there is none.
    """
    peaks = np.array(
        [p.idx for p in pivots if p.kind == +1], dtype=np.int64,
    )
    valleys = np.array(
        [p.idx for p in pivots if p.kind == -1], dtype=np.int64,
    )
    confirm = np.array([p.confirm_idx for p in pivots], dtype=np.int64)
    idxs = np.array([p.idx for p in pivots], dtype=np.int64)
    next_extreme = np.full(len(pivots), -1, dtype=np.int64)
    for k in range(len(pivots) - 1):
        next_extreme[k] = idxs[k + 1]
    return peaks, valleys, confirm, next_extreme
