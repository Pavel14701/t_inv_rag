"""Bar-by-bar DSL provider over a single OHLCV frame (TZ-05).

Prototype of the TZ-03 core: compute-once + cache + O(1) offset indexing.
Indicators are computed once for the whole series and cached by
(name, params). The key constraint: **causality** — every indicator in
the set (ema, sma, rsi, atr) depends only on bars <= t, so computing
over the full series is equivalent to computing over the [:t+1] slice
(look-ahead is impossible by construction).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from dsl.exceptions import ProviderError
from dsl.providers.base import IndicatorProvider


# Mapping of the unified schema (TZ-02) onto possible column names
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    'open': ('open', 'open_price'),
    'high': ('high', 'high_price'),
    'low': ('low', 'low_price'),
    'close': ('close', 'close_price'),
    'volume': ('volume',),
}


class WarmupNotReady(ProviderError):  # noqa: N818 - domain name is clearer
    """Indicator value is not ready yet (warm-up / insufficient history)."""


def _column(df: pl.DataFrame, name: str) -> pl.Series:
    for alias in _COLUMN_ALIASES[name]:
        if alias in df.columns:
            return df[alias]
    raise ValueError(
        f'column for {name!r} not found; tried {_COLUMN_ALIASES[name]}'
    )


def build_manifest() -> dict[str, Any]:
    """Manifest of series and indicators available at inference."""
    def _params(**kw):
        return {'parameters': kw} if kw else {}

    return {
        'indicators': {
            'close': {'attributes': []},
            'open': {'attributes': []},
            'high': {'attributes': []},
            'low': {'attributes': []},
            'volume': {'attributes': []},
            'ema': {
                'attributes': ['value'],
                'parameters': {'length': {'type': 'float', 'default': 20}},
            },
            'sma': {
                'attributes': ['value'],
                'parameters': {'length': {'type': 'float', 'default': 20}},
            },
            'rsi': {
                'attributes': ['value'],
                'parameters': {'length': {'type': 'float', 'default': 14}},
            },
            'atr': {
                'attributes': ['value'],
                'parameters': {'length': {'type': 'float', 'default': 14}},
            },
        },
    }


class BarSeriesProvider(IndicatorProvider):
    """Series provider: causal ta indicators over the full frame.

    Args:
        df: Polars DataFrame with OHLCV columns (unified names
            ``open/high/low/close/volume`` or legacy ``*_price``).
        ta: ta module (for lazy import of heavy computations).
        cursor: index of the "current" bar (0-based); advanced by the engine.

    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._n = len(df)
        self.cursor = self._n - 1
        self._series = {name: _column(df, name) for name in _COLUMN_ALIASES}
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._manifest = build_manifest()

    def get_manifest(self) -> dict[str, Any]:
        """Provider manifest."""
        return self._manifest

    @staticmethod
    def _ta_funcs() -> dict[str, Any]:
        """Lazy import of ta functions (numba/talib are heavy)."""
        from ta.src.momentum.rsi import rsi_ind
        from ta.src.overlap.ema import ema_ind
        from ta.src.overlap.sma import sma_ind
        from ta.src.volatility.atr import atr_ind

        return {
            'ema': ema_ind,
            'sma': sma_ind,
            'rsi': rsi_ind,
            'atr': atr_ind,
        }

    def _indicator_array(self, indicator: str, length: int) -> np.ndarray:
        key = (indicator, int(length))
        if key in self._cache:
            return self._cache[key]
        funcs = self._ta_funcs()
        close = self._series['close'].to_numpy()
        if indicator == 'ema':
            arr = np.asarray(funcs['ema'](close, length=length))
        elif indicator == 'sma':
            arr = np.asarray(funcs['sma'](close, length=length))
        elif indicator == 'rsi':
            arr = np.asarray(
                funcs['rsi'](close, length=length, nan_policy='ignore')
            )
        elif indicator == 'atr':
            arr = np.asarray(
                funcs['atr'](
                    self._series['high'].to_numpy(),
                    self._series['low'].to_numpy(),
                    close,
                    length=length,
                    nan_policy='ignore',
                )
            )
        else:  # pragma: no cover - manifest forbids other names
            raise ProviderError(f'unknown indicator {indicator!r}')
        arr = np.asarray(arr, dtype=np.float64).ravel()
        self._cache[key] = arr
        return arr

    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Value of a series/indicator at bar ``cursor - offset``.

        Raises:
            ProviderError: if the bar is out of range or the value is NaN
                (warm-up) — the engine classifies it as a warm-up skip.

        """
        idx = self.cursor - int(offset)
        if idx < 0 or idx >= self._n:
            raise WarmupNotReady(
                f'{indicator}: bar {idx} outside [0, {self._n})'
            )
        if indicator in self._series:
            value = float(self._series[indicator][idx])
        else:
            length = int(params.get('length', 20))
            if length < 1:
                raise ProviderError(f'{indicator}: length must be >= 1')
            if length > self._n:
                raise WarmupNotReady(
                    f'{indicator}: length {length} > bars {self._n}'
                )
            arr = self._indicator_array(indicator, length)
            value = float(arr[idx])
        if np.isnan(value):
            raise WarmupNotReady(
                f'{indicator}(length={params.get("length")}): warmup NaN '
                f'at bar {idx}'
            )
        return value