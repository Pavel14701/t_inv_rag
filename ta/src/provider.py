"""TaProvider: config-driven bridge between ta kernels and the DSL (TZ-03).

Key design decisions:
- **Offset semantics**: DSL ``x[n]`` means "n bars back" (cache index
  ``cursor - n``). The ta ``offset`` parameter is NEVER used (T1).
- **Compute-once**: each indicator is computed once over the full series,
  cached by ``(dsl_name, frozen params)``; ``resolve(offset)`` is O(1) (T2).
- **Multi-output**: indicators that return tuples map attributes to output
  indices (e.g. ott.direction -> output 3) via ``IndicatorBinding.outputs``
  (T3).
- **Warm-up contract**: bars < min_bars raise ``WarmupNotReady`` (skip);
  NaN after warm-up raises ``ProviderError`` (data anomaly) (T4).
"""

from __future__ import annotations

import inspect

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from dsl.exceptions import ProviderError
from dsl.providers.base import IndicatorProvider

from .candle.cdl_engulfing import cdl_engulfing
from .custom.ott import ott_ind
from .momentum.rsi import rsi_ind
from .overlap.ema import ema_ind
from .overlap.sma import sma_ind
from .statistics.entropy import entropy_ind
from .trend.adx import adx_ind
from .volatility.atr import atr_ind
from .volume.vwma import vwma_ind


__all__ = [
    "BINDINGS",
    "IndicatorBinding",
    "OutputSpec",
    "TaProvider",
    "WarmupNotReady",
    "build_manifest",
    "get_binding",
    "registered_names",
]


class WarmupNotReady(ProviderError):  # noqa: N818 - domain name is clearer
    """Indicator value not ready (warm-up / insufficient history)."""


# --------------------------------------------------------------------------- #
# 1. IndicatorBinding (TZ-03 item 3.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Maps a DSL attribute to a tuple element of the ta function output."""

    index: int  # 0 for single-output, 1+ for multi-output tuples


@dataclass(frozen=True, slots=True)
class IndicatorBinding:
    """Declarative bridge between a ta function and the DSL (TZ-03 п.3.1).

    Attributes:
        dsl_name: name used in DSL expressions (e.g. ``rsi``).
        func: the ta ``*_ind`` callable.
        sources: which price columns the function needs.
        outputs: maps DSL attribute name to output tuple index.
        default_params: defaults merged into user params.
        min_bars: warm-up bars as a function of resolved params.

    """

    dsl_name: str
    func: Callable[..., Any]
    sources: tuple[str, ...]
    outputs: dict[str, OutputSpec]
    default_params: dict[str, Any] = field(default_factory=dict)
    min_bars: Callable[[dict[str, Any]], int] = lambda p: int(
        p.get("length", 1)
    )


# --------------------------------------------------------------------------- #
# 2. Registry (TZ-03 items 3.1, 3.5)
# --------------------------------------------------------------------------- #

BINDINGS: dict[str, IndicatorBinding] = {}


def _register(binding: IndicatorBinding) -> IndicatorBinding:
    BINDINGS[binding.dsl_name] = binding
    return binding


_register(
    IndicatorBinding(
        dsl_name="ema",
        func=ema_ind,
        sources=("close",),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 20},
    )
)
_register(
    IndicatorBinding(
        dsl_name="sma",
        func=sma_ind,
        sources=("close",),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 20},
    )
)
_register(
    IndicatorBinding(
        dsl_name="rsi",
        func=rsi_ind,
        sources=("close",),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 14},
    )
)
_register(
    IndicatorBinding(
        dsl_name="adx",
        func=adx_ind,
        sources=("high", "low", "close"),
        outputs={
            "adx": OutputSpec(index=0),
            "plus_di": OutputSpec(index=1),
            "minus_di": OutputSpec(index=2),
        },
        default_params={"length": 14, "signal_length": 14},
        min_bars=lambda p: 2 * int(p.get("length", 14)),
    )
)
_register(
    IndicatorBinding(
        dsl_name="entropy",
        func=entropy_ind,
        sources=("close",),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 10},
    )
)
_register(
    IndicatorBinding(
        dsl_name="vwma",
        func=vwma_ind,
        sources=("close", "volume"),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 20},
    )
)
_register(
    IndicatorBinding(
        dsl_name="ott",
        func=ott_ind,
        sources=("close",),
        outputs={
            "ma": OutputSpec(index=0),
            "long_stop": OutputSpec(index=1),
            "short_stop": OutputSpec(index=2),
            "direction": OutputSpec(index=3),
            "ott": OutputSpec(index=4),
        },
        default_params={"length": 5, "percent": 2.0},
    )
)
# Price series bindings (direct column access, not computed indicators)
for _price_col in ("open", "high", "low", "close", "volume"):
    _register(
        IndicatorBinding(
            dsl_name=_price_col,
            func=lambda src, **kw: src,
            sources=(_price_col,),
            outputs={"value": OutputSpec(index=0)},
            min_bars=lambda p: 1,
        )
    )
_register(
    IndicatorBinding(
        dsl_name="engulfing",
        func=cdl_engulfing,
        sources=("open", "high", "low", "close"),
        outputs={"value": OutputSpec(index=0)},
        default_params={},
        min_bars=lambda p: 2,
    )
)
_register(
    IndicatorBinding(
        dsl_name="atr",
        func=atr_ind,
        sources=("high", "low", "close"),
        outputs={"value": OutputSpec(index=0)},
        default_params={"length": 14},
    )
)


def registered_names() -> set[str]:
    """Names available in DSL expressions."""
    return set(BINDINGS)


def get_binding(name: str) -> IndicatorBinding:
    """Resolve a binding by its DSL name."""
    return BINDINGS[name]


def build_manifest() -> dict[str, Any]:
    """Generate a DSL manifest from the binding registry (TZ-03 п.3.2)."""
    indicators: dict[str, Any] = {}
    for name, binding in BINDINGS.items():
        attrs = list(binding.outputs.keys())
        params = {}
        for key, val in binding.default_params.items():
            params[key] = {
                "type": "float" if isinstance(val, float) else "int",
                "default": val,
            }
        indicators[name] = {"attributes": attrs, "parameters": params}
    return {"indicators": indicators}


# --------------------------------------------------------------------------- #
# 3. TaProvider (TZ-03 item 3.3)
# --------------------------------------------------------------------------- #


class TaProvider(IndicatorProvider):
    """Compute-once indicator provider over a full OHLCV frame.

    Args:
        df: Polars DataFrame with columns matching every binding's
            ``sources`` (at minimum: ``close``; ATR also needs
            ``high``, ``low``).

    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._n = len(df)
        self.cursor = self._n - 1
        self._cache: dict[
            tuple[str, tuple[tuple[str, Any], ...]], np.ndarray
        ] = {}
        self._manifest = build_manifest()
        self._columns: dict[str, np.ndarray] = {}
        for binding in BINDINGS.values():
            for src in binding.sources:
                if src not in self._columns and src in df.columns:
                    self._columns[src] = df[src].to_numpy()

    def get_manifest(self) -> dict[str, Any]:
        """Return the DSL manifest generated from the binding registry."""
        return self._manifest

    def _indicator_array(
        self,
        binding: IndicatorBinding,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Compute-once: resolve the full indicator array (cached)."""
        length = int(
            params.get("length", binding.default_params.get("length", 1))
        )
        if length > self._n:
            raise WarmupNotReady(
                f"{binding.dsl_name}: length {length} > bars {self._n}"
            )
        kwargs: dict[str, Any] = {**binding.default_params, **params}
        kwargs["length"] = length
        kwargs.setdefault("nan_policy", "ignore")
        # filter to only params the function actually accepts
        valid = set(inspect.signature(binding.func).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
        # cache key must cover ALL resolved params, not just length:
        # macd(fast=12) vs macd(fast=26) must not collide (TZ-03 wave 2)
        key = (binding.dsl_name, tuple(sorted(kwargs.items())))
        if key in self._cache:
            return self._cache[key]
        args = [self._columns[src] for src in binding.sources]
        raw = binding.func(*args, **kwargs)
        if isinstance(raw, tuple):
            # multi-output: stack into (n_outputs, n_bars) matrix
            arr = np.asarray([np.asarray(r, dtype=np.float64) for r in raw])
        else:
            arr = np.asarray(raw, dtype=np.float64).ravel()
        self._cache[key] = arr
        return arr

    def resolve(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
        offset: int,
    ) -> float:
        """Value of ``indicator.attributes`` at bar ``cursor - offset``.

        Raises:
            WarmupNotReady: bar in warm-up period (T4).
            ProviderError: unknown indicator, NaN after warm-up.

        """
        binding = BINDINGS.get(indicator)
        if binding is None:
            raise ProviderError(f"unknown indicator {indicator!r}")

        idx = self.cursor - int(offset)
        if idx < 0 or idx >= self._n:
            raise WarmupNotReady(
                f"{indicator}: bar {idx} outside [0, {self._n})"
            )

        min_bars = binding.min_bars(params)
        if idx < min_bars:
            raise WarmupNotReady(
                f"{indicator}: warm-up (bar {idx} < min_bars {min_bars})"
            )

        arr = self._indicator_array(binding, params)
        attr = attributes[0] if attributes else "value"
        spec = binding.outputs.get(attr, OutputSpec(index=0))
        if arr.ndim == 2:
            value = float(arr[spec.index, idx])
        else:
            value = float(arr[idx])

        if np.isnan(value):
            raise WarmupNotReady(
                f"{indicator}({params}): warm-up NaN at bar {idx}"
            )
        return value


# TZ-03 wave 2: auto-derive bindings for every remaining ta indicator
# (manual bindings above are golden anchors and are never overwritten).
from . import registry as _registry  # noqa: E402


_registry.install_auto_bindings()
