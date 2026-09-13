"""Universal indicator mapper (TZ-03 wave 2).

Instead of hand-writing an :class:`IndicatorBinding` per indicator
(TZ-03 wave 1: 9 manual bindings as golden examples), bindings for the
remaining ~80 ``ta`` indicators are derived mechanically from the
``*_ind`` signatures:

- ``sources``       <- names of the data arguments (open/high/low/close/volume)
- ``default_params``<- signature defaults (numeric only; engine-handled
  args ``offset/fillna/nan_policy/trim/use_talib`` are excluded)
- ``outputs``       <- return annotation: ``np.ndarray`` -> {"value": 0};
  ``tuple[ndarray x N]`` -> N outputs; named when listed in
  :data:`NAMED_OUTPUTS`, otherwise generic ``out2..outN``
- ``min_bars``      <- max over ``*length`` params (overridable)

Facts that cannot be introspected live in the exception table
(:data:`NAMED_OUTPUTS`, :data:`MIN_BARS_OVERRIDES`, :data:`SKIP`).
Manual bindings registered in :mod:`ta.src.provider` win over derived
ones - they are the golden regression anchors.
"""

from __future__ import annotations

import inspect
import pkgutil

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import polars as pl

from .provider import BINDINGS, IndicatorBinding, OutputSpec, _register


__all__ = [
    "AUTO_REPORT",
    "MIN_BARS_OVERRIDES",
    "NAMED_OUTPUTS",
    "SKIP",
    "auto_bindings",
    "install_auto_bindings",
]

# Engine-handled universal args (TZ-03 п.3.1): never exposed as
# indicator params in the manifest.
_ENGINE_ARGS: frozenset[str] = frozenset(
    {"offset", "fillna", "nan_policy", "trim", "use_talib"}
)
_SOURCE_ARGS: frozenset[str] = frozenset(
    {"open", "open_", "high", "low", "close", "volume"}
)
_SOURCE_ALIASES: dict[str, str] = {"open_": "open"}


def _default_min_bars(params: dict[str, Any]) -> int:
    """Warm-up heuristic: the longest ``*length`` param (or 1)."""
    lengths = [
        int(v)
        for k, v in params.items()
        if k.endswith("length") and isinstance(v, (int, float))
    ]
    return max(lengths) if lengths else 1


# --------------------------------------------------------------------------- #
# Exception table: data, not code (TZ-11-style "config as data")
# --------------------------------------------------------------------------- #

#: Semantic names for tuple outputs (verified against docstrings).
NAMED_OUTPUTS: dict[str, dict[str, int]] = {
    "macd": {"macd": 0, "signal": 1, "hist": 2},
    "ppo": {"ppo": 0, "signal": 1, "hist": 2},
    "fisher": {"fisher": 0, "signal": 1},
    "brar": {"ar": 0, "br": 1},
    "kst": {"kst": 0, "signal": 1},
}

#: Non-standard warm-up (overrides the longest-length heuristic).
MIN_BARS_OVERRIDES: dict[str, Callable[[dict[str, Any]], int]] = {
    # coppock smooths the sum of several ROC windows
    "coppock": lambda p: 2
    * max(
        (
            int(v)
            for k, v in p.items()
            if k.endswith("length") and isinstance(v, (int, float))
        ),
        default=10,
    ),
}

#: Functions excluded from DSL exposure, with the reason.
SKIP: dict[str, str] = {
    "ichimoku": "takes a DataFrame, not per-column vectors",
    "scrsi": "mixed tuple output (arrays + scalars)",
    "zigzag": "variable-length output (pivots), engine needs fixed width",
    "tos_stdevall": "variable-length output, engine needs fixed width",
}

#: Bindings exist but the default-params engine path hits a known numba
#: dispatch bug (same as TZ-12 OTT note). Usable with explicit params.
SMOKE_SKIP: dict[str, str] = {
    "ott": "numba dispatch: length int64 vs compiled signature",
}


@dataclass(frozen=True, slots=True)
class _Derivation:
    """Introspection result for one ``*_ind`` function."""

    dsl_name: str
    sources: tuple[str, ...]
    default_params: dict[str, Any]
    width: int
    skip_reason: str = ""


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Type hints or empty dict when unresolvable (reported, not silent)."""
    try:
        return get_type_hints(fn)
    except (NameError, TypeError):
        return {}


def _derive(fn: Callable[..., Any]) -> _Derivation:
    """Derive binding facts from the function signature/annotation."""
    sig = inspect.signature(fn)
    hints = _resolve_hints(fn)

    sources: list[str] = []
    default_params: dict[str, Any] = {}
    required_extra = False
    for pname, param in sig.parameters.items():
        base = _SOURCE_ALIASES.get(pname, pname)
        if base in _SOURCE_ARGS:
            sources.append(base)
        if (
            pname not in _ENGINE_ARGS
            and param.default is not inspect.Parameter.empty
            and isinstance(param.default, (int, float))
            and not isinstance(param.default, bool)
        ):
            default_params[pname] = param.default
        if (
            pname not in _ENGINE_ARGS
            and base not in _SOURCE_ARGS
            and param.default is inspect.Parameter.empty
        ):
            # required non-data arg without default: engine can only pass
            # series + defaults, so such an indicator needs explicit params
            required_extra = True

    def skip(reason: str) -> _Derivation:
        return _Derivation("", tuple(sources), default_params, 0, reason)

    if required_extra:
        return skip("requires explicit params (required args w/o defaults)")
    ret = hints.get("return")
    if ret is None:
        return skip("no resolvable return annotation")
    for hint in hints.values():
        if hint is pl.DataFrame:
            return skip("DataFrame-based function")
    if get_origin(ret) is tuple:
        args = get_args(ret)
        if not args or not all(a is np.ndarray for a in args):
            return skip("mixed tuple output")
        width = len(args)
    elif ret is np.ndarray or (
        get_origin(ret) is not None and np.ndarray in get_args(ret)
    ):
        width = 1
    else:
        return skip("unsupported return annotation")
    return _Derivation("", tuple(sources), default_params, width)


def _outputs_for(dsl_name: str, width: int) -> dict[str, OutputSpec]:
    """Named or generic attribute -> tuple index mapping."""
    named = NAMED_OUTPUTS.get(dsl_name)
    if named is not None and len(named) == width:
        return {k: OutputSpec(index=v) for k, v in named.items()}
    outputs = {"value": OutputSpec(index=0)}
    for i in range(1, width):
        outputs[f"out{i + 1}"] = OutputSpec(index=i)
    return outputs


def _iter_indicator_funcs() -> list[tuple[str, Callable[..., Any]]]:
    """Collect module-level ``*_ind`` functions across ``ta.src``."""
    import ta.src as pkg

    found: list[tuple[str, Callable[..., Any]]] = []
    for mod_info in pkgutil.walk_packages(pkg.__path__, prefix="ta.src."):
        if mod_info.ispkg:
            continue
        module = import_module(mod_info.name)
        for name, obj in vars(module).items():
            if (
                callable(obj)
                and name.endswith("_ind")
                and len(name) > 4
                and getattr(obj, "__module__", None) == mod_info.name
            ):
                found.append((name[:-4], obj))
    return found


AUTO_REPORT: dict[str, str] = {}


def auto_bindings() -> dict[str, IndicatorBinding]:
    """Derive bindings for every ``*_ind`` not yet registered.

    Returns only NEW bindings (existing manual entries win). Skipped
    functions are recorded in :data:`AUTO_REPORT` with the reason.
    """
    new: dict[str, IndicatorBinding] = {}
    for dsl_name, fn in _iter_indicator_funcs():
        if dsl_name in SKIP:
            AUTO_REPORT[dsl_name] = SKIP[dsl_name]
            continue
        if dsl_name in BINDINGS:
            AUTO_REPORT[dsl_name] = "manual binding (golden anchor)"
            continue
        derived = _derive(fn)
        if derived.skip_reason:
            AUTO_REPORT[dsl_name] = derived.skip_reason
            continue
        new[dsl_name] = IndicatorBinding(
            dsl_name=dsl_name,
            func=fn,
            sources=derived.sources,
            outputs=_outputs_for(dsl_name, derived.width),
            default_params=derived.default_params,
            min_bars=MIN_BARS_OVERRIDES.get(dsl_name, _default_min_bars),
        )
    return new


def install_auto_bindings() -> int:
    """Register derived bindings; returns how many were added."""
    added = 0
    for binding in auto_bindings().values():
        _register(binding)
        added += 1
    return added
