# _imports.py
import importlib
import warnings
from types import ModuleType
from typing import Any, Tuple

_cache: dict[str, tuple[ModuleType | None, bool]] = {}


def import_lib(name: str, warning_msg: str | None = None) -> Tuple[Any, bool]:
    """
    
    Imports a library by name and returns a tuple (module, available).
    If the library is missing, module = None, available = False.
    A warning is only issued the first time a missing library is detected.
    """
    if name in _cache:
        return _cache[name]

    try:
        module = importlib.import_module(name)
        available = True
    except ImportError:
        module = None
        available = False
        if warning_msg:
            warnings.warn(warning_msg)

    _cache[name] = (module, available)
    return module, available