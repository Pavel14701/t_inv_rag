"""Rule registry for the config-driven risk engine (TZ-11 item 3.3).

Every rule is a "contract": ``name`` selects an implementation from the
registry, ``params`` are plain data. The engine holds no rule logic.
New rules are added as code in :mod:`risk.rules.impl` and enabled or
disabled via config.
"""

from .base import RuleResult, RuleState, get, register, registered_names


__all__ = [
    "RuleResult",
    "RuleState",
    "get",
    "register",
    "registered_names",
]

# Eagerly import concrete rules so the registry is populated on package
# import (both for the engine and for config validation).
from . import impl as _impl  # noqa: F401
