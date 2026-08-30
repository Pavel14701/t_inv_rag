"""External library loader for TA-Lib and yfinance.

This module centralises all imports of optional external libraries.
It is safe to import from anywhere in the package without causing cycles.
"""

import warnings
from types import ModuleType
from typing import Any

_cache: dict[str, tuple[ModuleType | None, bool]] = {}


def _import_lib(name: str, warning_msg: str | None = None) -> tuple[Any, bool]:
    """Import a library by name, return (module, available)."""
    if name in _cache:
        return _cache[name]
    try:
        module = __import__(name)
        available = True
    except ImportError:
        module = None
        available = False
        if warning_msg:
            warnings.warn(warning_msg)
    _cache[name] = (module, available)
    return module, available


# --- TA-Lib ---
talib, talib_available = _import_lib(
    'talib',
    'TA-Lib not installed. Using Numba fallback.'
)

# --- yfinance ---
yfinance, yfinance_available = _import_lib(
    'yfinance',
    "yfinance not installed. Please install via 'pip install yfinance'"
)

# --- TA-Lib MA type mapping (only if available) ---
_TALIB_MA_MAP = {
    'sma': talib.MA_Type.SMA if talib_available else None,
    'ema': talib.MA_Type.EMA if talib_available else None,
    'wma': talib.MA_Type.WMA if talib_available else None,
    'dema': talib.MA_Type.DEMA if talib_available else None,
    'tema': talib.MA_Type.TEMA if talib_available else None,
    'trima': talib.MA_Type.TRIMA if talib_available else None,
    'kama': talib.MA_Type.KAMA if talib_available else None,
    'mama': talib.MA_Type.MAMA if talib_available else None,
    't3': talib.MA_Type.T3 if talib_available else None,
}

__all__ = [
    'talib',
    'talib_available',
    'yfinance',
    'yfinance_available',
    '_TALIB_MA_MAP',
]
