"""Bar-by-bar DSL provider over a single OHLCV frame (TZ-05).

Прототип ядра TZ-03: compute-once + кэш + O(1) offset-индексация.
Индикаторы считаются один раз на весь ряд и кэшируются по
(name, params). Ключевое ограничение: **каузальность** — все
индикаторы набора (ema, sma, rsi, atr) зависят только от баров
<= t, поэтому вычисление на полном ряду эквивалентно вычислению
на срезе [:t+1] (look-ahead невозможен по построению).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from dsl.exceptions import ProviderError
from dsl.providers.base import IndicatorProvider

# Маппинг унифицированной схемы (TZ-02) на возможные имена колонок
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    'open': ('open', 'open_price'),
    'high': ('high', 'high_price'),
    'low': ('low', 'low_price'),
    'close': ('close', 'close_price'),
    'volume': ('volume',),
}


class WarmupNotReady(ProviderError):  # noqa: N818 - доменное имя осмысленнее
    """Значение индикатора ещё не готово (прогрев / нехватка истории)."""


def _column(df: pl.DataFrame, name: str) -> pl.Series:
    for alias in _COLUMN_ALIASES[name]:
        if alias in df.columns:
            return df[alias]
    raise ValueError(
        f'column for {name!r} not found; tried {_COLUMN_ALIASES[name]}'
    )


def build_manifest() -> dict[str, Any]:
    """Манифест доступных в инференсе серий и индикаторов."""
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
    """Провайдер-ряд: каузальные индикаторы ta на полном кадре.

    Args:
        df: Polars DataFrame с колонками OHLCV (унифицированные имена
            ``open/high/low/close/volume`` или legacy ``*_price``).
        ta: Модуль ta (для ленивого импорта тяжёлых вычислений).
        cursor: Индекс «текущего» бара (0-based); двигается движком.

    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._n = len(df)
        self.cursor = self._n - 1
        self._series = {name: _column(df, name) for name in _COLUMN_ALIASES}
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._manifest = build_manifest()

    def get_manifest(self) -> dict[str, Any]:
        """Манифест провайдера."""
        return self._manifest

    @staticmethod
    def _ta_funcs() -> dict[str, Any]:
        """Ленивый импорт ta-функций (numba/talib тяжёлые)."""
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
        else:  # pragma: no cover - манифест запрещает прочие имена
            raise ProviderError(f'unknown indicator {indicator!r}')
        arr = np.asarray(arr, dtype=np.float64).ravel()
        self._cache[key] = arr
        return arr

    def resolve(  # noqa: C901 - единая диспетчеризация простая
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Значение серии/индикатора на баре ``cursor - offset``.

        Raises:
            ProviderError: если бар вне диапазона или значение NaN
                (прогрев) — движок классифицирует как warmup-пропуск.

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