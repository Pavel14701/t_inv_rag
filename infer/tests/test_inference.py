"""Tests for TZ-05 inference (provider, engine, CLI smoke)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from infer.cli import main as cli_main
from infer.data import load_synthetic, normalize
from infer.engine import run_inference
from infer.provider import BarSeriesProvider, WarmupNotReady


DSL_ENTRY = "close > ema(length=20)"


def test_provider_matches_ta_directly():
    """Parity: resolve(t) == the ta function at bar t (TZ-05 acceptance)."""
    df = load_synthetic(n_bars=300)
    provider = BarSeriesProvider(df)
    from ta.src.overlap.ema import ema_ind

    expected = np.asarray(ema_ind(df["close"].to_numpy(), length=20))
    for t in (50, 150, 299):
        provider.cursor = t
        got = provider.resolve("ema", {"length": 20}, [], 0)
        assert got == pytest.approx(float(expected[t]))


def test_provider_offset_semantics():
    """DSL offset [n] means n bars back, not a ta-offset (TZ-03 T1)."""
    df = load_synthetic(n_bars=100)
    provider = BarSeriesProvider(df)
    provider.cursor = 50
    close_now = provider.resolve("close", {}, [], 0)
    close_prev = provider.resolve("close", {}, [], 3)
    assert close_prev == pytest.approx(float(df["close"][47]))
    assert close_now == pytest.approx(float(df["close"][50]))


def test_provider_warmup_raises():
    """Before warm-up an indicator raises WarmupNotReady, not NaN."""
    df = load_synthetic(n_bars=50)
    provider = BarSeriesProvider(df)
    provider.cursor = 5
    with pytest.raises(WarmupNotReady):
        provider.resolve("ema", {"length": 20}, [], 0)


def test_normalize_legacy_schema():
    """Legacy PriceDataFramePolars schema normalizes to unified."""
    legacy = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open_price": [1.0, 2.0],
            "close_price": [1.5, 2.5],
            "high_price": [1.6, 2.6],
            "low_price": [0.9, 1.9],
            "volume": [10, 20],
            "turnover": [15.0, 25.0],
        }
    )
    df = normalize(legacy)
    for col in ("open", "high", "low", "close", "volume", "date"):
        assert col in df.columns


def test_run_inference_signals_and_summary():
    """End-to-end: signals, summary, warm-up skips (TZ-05 item 3.3)."""
    df = load_synthetic(n_bars=500)
    result = run_inference(df, DSL_ENTRY, "close < ema(length=20)")
    assert result.num_bars == 500
    assert result.num_entry > 0
    assert result.num_warmup_skips >= 19  # ema(20) warm-up
    assert result.signals.height == 500
    assert set(result.signals.columns) == {
        "date",
        "entry_signal",
        "exit_signal",
        "p_win",
    }
    # p_win without --ml is NaN
    assert result.signals["p_win"].is_nan().all()


def test_run_inference_parse_error_early():
    """A DSL syntax error fails before the run."""
    df = load_synthetic(n_bars=50)
    with pytest.raises(Exception):
        run_inference(df, "close >")


def test_cli_synthetic_smoke(tmp_path, capsys):
    """CLI smoke on synthetic data: signal file created, exit code 0."""
    out = tmp_path / "signals.csv"
    code = cli_main(
        [
            "--source",
            "synthetic",
            "--bars",
            "300",
            "--entry",
            DSL_ENTRY,
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    signals = pl.read_csv(out)
    assert signals.height == 300
    assert "entry_signal" in signals.columns
    captured = capsys.readouterr()
    assert "bars=300" in captured.out


def test_predict_p_win_at_price_feature_mismatch():
    """A bundle with n_price_feats != 5 raises a clear DSLError."""
    from dsl.exceptions import DSLError
    from infer.engine import predict_p_win_at

    class _Bundle:
        seq_len = 8
        model_config = {
            "n_price_feats": 4,
            "n_ind_feats": 0,
            "n_sig_feats": 0,
            "n_tp_sl_feats": 0,
        }

    class _Predictor:
        bundle = _Bundle()

    df = load_synthetic(n_bars=50)
    with pytest.raises(DSLError, match="price feature count mismatch"):
        predict_p_win_at(_Predictor(), df, t=20)


def test_predict_p_win_at_matching_bundle():
    """Happy path: a 5-price-feature bundle returns P(win)."""
    import torch  # noqa: F401 - guarantees ai dependencies are present

    from ai.src.bundle import EntryExitPredictor, build_bundle
    from ai.src.transformer import EntryExitTransformer
    from infer.engine import predict_p_win_at

    model_cfg = {
        "n_price_feats": 5,
        "n_ind_feats": 0,
        "n_sig_feats": 0,
        "n_tp_sl_feats": 0,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "max_seq_len": 64,
        "max_ob_seq_len": 8,
        "n_patterns": 2,
        "ob_embedding_dim": 4,
    }
    model = EntryExitTransformer(**model_cfg)
    bundle = build_bundle(model, model_config=model_cfg, seq_len=8)
    predictor = EntryExitPredictor(bundle)

    df = load_synthetic(n_bars=50)
    p_win = predict_p_win_at(predictor, df, t=20)
    assert p_win is not None
    assert 0.0 <= p_win <= 1.0
