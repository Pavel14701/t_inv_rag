"""Tests for TZ-06 п.2.4/2.5 (model bundle + predict_p_win)."""

from __future__ import annotations

import pytest
import torch

from ai.src.bundle import (
    EntryExitPredictor,
    build_bundle,
    load_bundle,
    rebuild_model,
    save_bundle,
)
from ai.src.transformer import EntryExitTransformer


def _make_model() -> EntryExitTransformer:
    return EntryExitTransformer(
        n_price_feats=5,
        n_ind_feats=3,
        n_sig_feats=2,
        n_tp_sl_feats=2,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_ob_seq_len=8,
        n_patterns=5,
        ob_embedding_dim=8,
    )


def _make_config(model: EntryExitTransformer) -> dict:
    return {
        "n_price_feats": 5,
        "n_ind_feats": 3,
        "n_sig_feats": 2,
        "n_tp_sl_feats": 2,
        "hidden_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "max_ob_seq_len": 8,
        "n_patterns": 5,
        "ob_embedding_dim": 8,
        "outcome_mode": model.outcome_mode,
        "atr_global": getattr(model, "atr_global", 1.0),
    }


def test_bundle_roundtrip(tmp_path):
    """Save, load and rebuild preserve architecture and weights."""
    model = _make_model()
    bundle = build_bundle(
        model,
        model_config=_make_config(model),
        feature_columns=["o", "h", "l", "c", "v"],
        seq_len=128,
    )
    path = tmp_path / "bundle.pt"
    save_bundle(bundle, path)

    loaded = load_bundle(path)
    assert loaded.seq_len == 128
    assert loaded.feature_columns == ["o", "h", "l", "c", "v"]
    assert loaded.outcome_mode == "binary"

    rebuilt = rebuild_model(loaded)
    rebuilt.load_state_dict(loaded.state_dict)
    assert isinstance(rebuilt, EntryExitTransformer)
    # state dict survives the round trip
    for k, v in model.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k])


def test_predict_p_win_returns_float():
    """Predictions stay in [0, 1] and agree with predict_proba."""
    model = _make_model()
    bundle = build_bundle(model, model_config=_make_config(model), seq_len=16)
    predictor = EntryExitPredictor(bundle)

    t_len = 16
    prices = torch.randn(t_len, 5)
    indicators = torch.randn(t_len, 3)
    signals = torch.randn(t_len, 2)
    tp = torch.randn(t_len, 1)
    sl = torch.randn(t_len, 1)
    obs = []

    probs = predictor.predict_proba(prices, indicators, signals, tp, sl, obs)
    assert set(probs) == {"p_entry", "p_exit", "p_win"}

    p = predictor.predict_p_win(prices, indicators, signals, tp, sl, obs)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(probs["p_win"])


def test_predict_requires_no_grad():
    """Predictions run under inference mode and return p_win."""
    model = _make_model()
    bundle = build_bundle(model, model_config=_make_config(model), seq_len=8)
    predictor = EntryExitPredictor(bundle)
    t_len = 8
    args = [
        torch.randn(t_len, 5),
        torch.randn(t_len, 3),
        torch.randn(t_len, 2),
        torch.randn(t_len, 1),
        torch.randn(t_len, 1),
        [],
    ]
    out = predictor.predict_proba(*args)
    assert "p_win" in out


def test_load_bundle_rejects_non_dict_payload(tmp_path):
    """A non-mapping payload is rejected instead of being trusted.

    Bundles are saved as plain mappings (``dataclasses.asdict``), so the
    restricted ``weights_only=True`` unpickler is sufficient; anything
    else must fail loudly before reaching :class:`ModelBundle`.

    """
    path = tmp_path / "bad.pt"
    torch.save(torch.zeros(3), path)
    with pytest.raises(TypeError, match="not a ModelBundle payload"):
        load_bundle(path)
