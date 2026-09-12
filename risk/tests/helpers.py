"""Shared helpers for risk tests (kept out of conftest for reuse)."""

from __future__ import annotations

import copy


def build(overrides: dict | None = None) -> dict:
    """Build a valid risk config dict; merge ``overrides`` on top."""
    cfg = {
        "engine": {
            "rules": [
                {
                    "name": "require_stop_loss",
                    "active": True,
                    "params": {
                        "sl": {"min_mult": 0.0},
                        "tp": {"min_mult": 0.1},
                    },
                },
                {"name": "position_limit", "active": True, "params": {}},
                {
                    "name": "max_positions",
                    "active": True,
                    "params": {"max_open": 5},
                },
                {
                    "name": "daily_loss_limit",
                    "active": True,
                    "params": {"max_daily_loss_pct": 0.02},
                },
                {
                    "name": "drawdown_stop",
                    "active": True,
                    "params": {"max_drawdown_pct": 0.2, "pause_bars": 60},
                },
            ]
        },
        "portfolio": {
            "max_capital_pct": 0.1,
            "max_units": None,
            "instruments": {},
        },
    }
    merged = copy.deepcopy(cfg)
    if overrides:
        merged["engine"] = overrides.get("engine", merged["engine"])
        merged["portfolio"] = overrides.get("portfolio", merged["portfolio"])
    return merged
