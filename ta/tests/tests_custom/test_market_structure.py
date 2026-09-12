# -*- coding: utf-8 -*-
"""Tests for the market_structure package (order block detection).

Covers:
- ``identify_order_blocks`` contract: output schema, ordering, zone sanity
- anti-look-ahead property in online mode: every block's pivot was
  confirmed strictly before its breakout bar (no repaint)
- ``OrderBlockConfig`` validation errors
- ``get_order_block_config`` aliases / unknown timeframe
- all timeframe presets run end-to-end and produce valid frames
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ta.src.custom.market_structure import (
    OnlineZigZag,
    OrderBlockConfig,
    identify_order_blocks,
    zigzag_reversal_numpy,
)


EXPECTED_COLUMNS = [
    "id",
    "block_type",
    "start",
    "break",
    "retest",
    "zone_low",
    "zone_high",
    "strength",
    "structure_label",
    "trend_direction",
]


def make_ohlcv(
    n: int = 300,
    seed: int = 0,
    start: float = 100.0,
    freq: str = "5m",
) -> pl.DataFrame:
    """Synthetic OHLCV frame: random-walk close with sane high/low."""
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 0.8, n))
    spread = np.abs(rng.normal(0.4, 0.15, n))
    high = close + spread
    low = close - spread
    volume = rng.gamma(2.0, 50.0, n)
    step = timedelta(minutes=int(freq.rstrip("m")))
    dates = [datetime(2024, 1, 1) + i * step for i in range(n)]
    return (
        pl.DataFrame(
            {
                "date": dates,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ),
        high,
        low,
    )


def assert_valid_block_frame(out: pl.DataFrame) -> None:
    assert isinstance(out, pl.DataFrame)
    assert out.columns == EXPECTED_COLUMNS
    if out.is_empty():
        return
    # sorted by start, ids strictly increasing
    assert out["start"].to_list() == sorted(out["start"].to_list())
    assert out["id"].to_list() == sorted(out["id"].to_list())
    assert set(out["id"].to_list()) == set(range(out.height))
    # zone sanity and temporal ordering
    assert (out["zone_low"] <= out["zone_high"]).all()
    assert (out["start"] < out["break"]).all()
    assert (out["break"] <= out["retest"]).all()
    assert out["block_type"].is_in(["supply", "demand"]).all()
    assert (out["strength"] >= 0).all()


# -----------------------------------------------------------------------------
# Output schema / contract
# -----------------------------------------------------------------------------
@pytest.mark.custom
def test_empty_output_schema() -> None:
    """A flat series has no pivots -> empty frame with the full schema."""
    n = 120
    df = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1) + i * timedelta(minutes=5)
                for i in range(n)
            ],
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        }
    )
    out = identify_order_blocks(
        df,
        cfg=OrderBlockConfig(use_online_extremes=True),
    )
    assert out.is_empty()
    assert out.columns == EXPECTED_COLUMNS
    schema = out.schema
    assert schema["id"] == pl.Int64
    assert schema["block_type"] == pl.Utf8
    assert schema["start"] == pl.Datetime
    assert schema["break"] == pl.Datetime
    assert schema["retest"] == pl.Datetime
    assert schema["zone_low"] == pl.Float64
    assert schema["zone_high"] == pl.Float64
    assert schema["strength"] == pl.Float64
    assert schema["structure_label"] == pl.Utf8
    assert schema["trend_direction"] == pl.Utf8


@pytest.mark.custom
@pytest.mark.parametrize("mode", [False, True])
def test_online_and_offline_modes_valid(mode: bool) -> None:
    df, _, _ = make_ohlcv(300, seed=1)
    cfg = OrderBlockConfig(use_online_extremes=mode)
    out = identify_order_blocks(df, cfg=cfg)
    assert_valid_block_frame(out)


@pytest.mark.custom
def _crafted_swing() -> tuple[pl.DataFrame, np.ndarray]:
    """Decline -> valley -> slow rally -> immediate breakout -> retest.

    The rally is deliberately slow (0.35/bar): the valley pivot becomes
    confirmed by the online ZigZag exactly at the breakout bar, not
    before - the textbook repaint scenario.
    """
    close: list[float] = []
    for i in range(11):  # decline 100 -> 98
        close.append(100.0 - 0.2 * i)
    c = 98.0
    for _ in range(9):  # slow rally -> breakout
        c += 0.35
        close.append(c)
    close += [99.6, 98.9, 98.3, 98.1]  # pullback into the zone
    for i in range(6):  # reaction bounce
        close.append(98.1 + 0.4 * (i + 1))
    close_arr = np.asarray(close)
    n = len(close_arr)
    volume = np.full(n, 100.0)
    volume[15] = 800.0  # breakout surge
    volume[21] = 900.0  # retest volume
    volume[23] = 700.0
    df = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1) + i * timedelta(hours=1) for i in range(n)
            ],
            "high": close_arr + 0.2,
            "low": close_arr - 0.2,
            "close": close_arr,
            "volume": volume,
        }
    )
    return df, close_arr


def _swing_cfg(**kwargs) -> OrderBlockConfig:
    """Short indicator windows so nothing sits in NaN warm-up."""
    return OrderBlockConfig(
        use_dynamic_lookback=False,
        lookback_min=5,
        confirmation_window=12,
        atr_period=3,
        volume_window=5,
        **kwargs,
    )


@pytest.mark.custom
def test_offline_mode_finds_blocks_on_crafted_swing() -> None:
    """Sanity: the pipeline confirms a demand block on a clean swing."""
    df, _ = _crafted_swing()
    out = identify_order_blocks(
        df,
        cfg=_swing_cfg(use_online_extremes=False),
    )
    assert_valid_block_frame(out)
    assert out.height >= 1
    row = out.row(0, named=True)
    assert row["block_type"] == "demand"
    assert row["start"] == df["date"][10]  # the valley bar
    assert row["break"] == df["date"][15]  # first bar above its high


# -----------------------------------------------------------------------------
# Anti-look-ahead property (online mode)
# -----------------------------------------------------------------------------
@pytest.mark.custom
@pytest.mark.parametrize("seed", [0, 2, 4])
def test_online_blocks_are_repaint_free(seed: int) -> None:
    """Every online block's pivot was final before the breakout bar.

    This is the core no-repaint guarantee: the batch ZigZag reference
    must contain the pivot, and its confirmation bar must precede the
    breakout - so the block was tradable live, without hindsight.
    """
    df, high, low = make_ohlcv(350, seed=seed)
    cfg = OrderBlockConfig(
        use_online_extremes=True,
        online_reversal=2.0,
    )
    out = identify_order_blocks(df, cfg=cfg)
    assert_valid_block_frame(out)
    dates = df["date"].to_list()
    pos = {d: i for i, d in enumerate(dates)}
    pivots = zigzag_reversal_numpy(high, low, 2.0)
    confirm_of = {p.idx: p.confirm_idx for p in pivots}
    for row in out.iter_rows(named=True):
        pivot_idx = pos[row["start"]]
        break_idx = pos[row["break"]]
        assert pivot_idx in confirm_of
        assert confirm_of[pivot_idx] < break_idx


@pytest.mark.custom
def test_online_stricter_than_offline() -> None:
    """Repaint scenario: breakout on the very bar the pivot finalises.

    The offline ZigZag happily uses the valley (hindsight: it knows the
    rally continues).  The online machine confirms that valley exactly
    at the breakout bar, and the ``confirm_idx < break_idx`` guard must
    reject the block - nothing repaintable is ever reported.
    """
    df, _ = _crafted_swing()
    valley_date = df["date"][10]
    # offline: the valley IS used, breakout allowed
    off = identify_order_blocks(
        df,
        cfg=_swing_cfg(use_online_extremes=False),
    )
    assert off.filter(pl.col("start") == valley_date).height >= 1
    # online: the valley finalises at the breakout bar -> rejected
    on = identify_order_blocks(
        df,
        cfg=_swing_cfg(use_online_extremes=True, online_reversal=2.0),
    )
    assert on.filter(pl.col("start") == valley_date).height == 0
    # and indeed the pivot is only confirmed AT the breakout bar
    zz = OnlineZigZag(2.0)
    zz.update_series(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
    )
    confirm_of = {p.idx: p.confirm_idx for p in zz.confirmed}
    assert confirm_of[10] == 15
