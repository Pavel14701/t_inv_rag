"""Tests for the YAML configuration layer (TZ-06 п.10)."""

from ..config import (  # noqa: TID100 - package-relative for pytest isolation
    AIConfig,
    RiskConfig,
    load_config,
    risk_kwargs,
    set_seed,
)


def test_defaults_equal_legacy_hardcode():
    """YAML defaults must equal the former hardcoded constants."""
    cfg = AIConfig()
    r = cfg.risk
    assert (r.atr_period, r.atr_floor) == (14, 1.0e-6)
    assert (r.tp_atr_multiplier, r.sl_atr_multiplier) == (2.0, 1.5)
    assert abs(r.min_rr - 1 / 3) < 1e-4
    assert (r.commission_pct, r.slippage_pct, r.max_bars_hold) == (
        0.001, 0.0005, 20,
    )
    assert r.use_r_multiple is False
    m = cfg.model
    assert (m.seq_len, m.hidden_size, m.num_layers, m.num_heads) == (
        128, 128, 4, 8,
    )
    assert m.close_idx == 3
    t = cfg.training
    assert (t.epochs, t.batch_size, t.lr) == (10, 16, 1.0e-4)
    assert (t.lambda_outcome, t.lambda_pattern) == (0.3, 0.1)
    assert (t.val_split, t.patience) == (0.2, 3)
    assert (t.num_rounds, t.action_threshold, t.outcome_threshold) == (
        3, 0.9, 0.8,
    )


def test_load_from_yaml_overrides(tmp_path):
    """Unknown keys are ignored; known keys override defaults."""
    yml = tmp_path / 'ai.yaml'
    yml.write_text(
        'seed: 7\n'
        'risk:\n'
        '  tp_atr_multiplier: 3.0\n'
        '  unknown_key: 1\n'
        'compute:\n'
        '  train_backend: cpu\n',
        encoding='utf-8',
    )
    cfg = load_config(yml)
    assert cfg.seed == 7
    assert cfg.risk.tp_atr_multiplier == 3.0
    assert cfg.risk.sl_atr_multiplier == 1.5  # default preserved
    assert cfg.compute.train_backend == 'cpu'


def test_missing_file_returns_defaults(tmp_path):
    """Absent YAML yields pure defaults."""
    cfg = load_config(tmp_path / 'nope.yaml')
    assert isinstance(cfg, AIConfig)
    assert cfg.seed == 42


def test_risk_kwargs_mapping():
    """Mapping preserves config values into kwargs dict."""
    kw = risk_kwargs(RiskConfig(max_bars_hold=5))
    assert kw['max_bars_hold'] == 5
    assert kw['commission_pct'] == 0.001


def test_set_seed_reproducible():
    """Same seed produces identical numpy sequence."""
    import numpy as np

    set_seed(123)
    a = np.random.rand()
    set_seed(123)
    b = np.random.rand()
    assert a == b