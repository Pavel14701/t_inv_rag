# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING

from ._imports import import_lib

# Получаем модули и флаги доступности
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

__all__ = [
    "talib", "talib_available",
    "yfinance", "yfinance_available",
]