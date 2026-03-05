# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING

from ._imports import import_lib
from .ma import ma_mode

talib_module, talib_available = import_lib(
    "talib", "TA-Lib not installed. Using Numba fallback."
)
yfinance_module, yfinance_available = import_lib(
    "yfinance", "yfinance not installed. Please install via 'pip install yfinance'"
)

if TYPE_CHECKING:
    import talib as _talib
    import yfinance as _yfinance
    talib = _talib
    yfinance = _yfinance
else:
    talib = talib_module
    yfinance = yfinance_module

_TALIB_MA_MAP = {
    "sma": talib.MA_Type.SMA if talib_available else None,
    "ema": talib.MA_Type.EMA if talib_available else None,
    "wma": talib.MA_Type.WMA if talib_available else None,
    "dema": talib.MA_Type.DEMA if talib_available else None,
    "tema": talib.MA_Type.TEMA if talib_available else None,
    "trima": talib.MA_Type.TRIMA if talib_available else None,
    "kama": talib.MA_Type.KAMA if talib_available else None,
    "mama": talib.MA_Type.MAMA if talib_available else None,
    "t3": talib.MA_Type.T3 if talib_available else None,
}


__all__ = [
    "talib", "talib_available",
    "yfinance", "yfinance_available",
    "ma_mode",
    "_TALIB_MA_MAP",
]

