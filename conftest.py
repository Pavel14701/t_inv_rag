"""Root pytest configuration.

Applies per-service markers so `-m dsl`, `-m ta`, ... work from anywhere
(TZ-14 п.2.3). Markers are registered once at the root pyproject and here
mapped onto collection paths; tests must NOT mark themselves manually.
"""

from __future__ import annotations

import pathlib

import pytest


def pytest_collection_modifyitems(items) -> None:
    """Tag every test with the marker of its owning package."""
    root = pathlib.Path(__file__).resolve().parent
    services = {
        "dsl": "dsl",
        "ta": "ta",
        "ai": "ai",
        "infer": "infer",
        "rag": "rag",
        "risk": "risk",
        "main": "main",
        "strategies": "strategies",
    }
    for item in items:
        try:
            rel = pathlib.Path(str(item.path)).resolve().relative_to(root)
        except ValueError:
            continue
        for svc, marker in services.items():
            if svc in {part for part in rel.parts}:
                item.add_marker(getattr(pytest.mark, marker))
                break
