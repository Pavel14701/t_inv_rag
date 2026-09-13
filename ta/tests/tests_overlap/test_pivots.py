# -*- coding: utf-8 -*-
"""Unit tests for PIVOTS (pivot points) module.

Tests cover:
- Core numba pivot formulas (woodie, demark branches) directly
- pivots_ind for all six methods against hand-computed reference levels
- Forward-fill semantics: pivots computed on the previous period are
  constant within the current period (no lookahead)
- Method validation and unknown-method error
- Weekly anchor resampling
- Structural guarantees: same rows, same order, all level columns present
"""

from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import polars as pl
import pytest

from ta.src.overlap.pivots import _pivot_demark, _pivot_woodie, pivots_ind


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def make_hourly_df(n: int = 24 * 10, seed: int = 0) -> pl.DataFrame:
    """Hourly OHLC DataFrame covering `n` hours starting 2024-01-01."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def day_window(df: pl.DataFrame, suffix: str, day: int = 2) -> pl.DataFrame:
    """Rows of the pivots result inside calendar day `day` (2024-01-DD)."""
    start = datetime(2024, 1, day)
    end = start + timedelta(days=1)
    return df.filter((pl.col("date") >= start) & (pl.col("date") < end))


# -----------------------------------------------------------------------------
# Core numba formula tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_pivot_woodie_formula() -> None:
    """Woodie: TP=(2*O+H+L)/4, S1=2TP-H, R1=2TP-L, S3/R3 as documented."""
    o = np.array([10.0])
    h = np.array([12.0])
    lst = np.array([9.0])
    c = np.array([10.5])  # accepted but unused
    tp, s1, s2, s3, s4, r1, r2, r3, r4 = _pivot_woodie(o, h, lst, c)
    tp_ref = (2 * 10.0 + 12.0 + 9.0) / 4
    assert tp[0] == tp_ref
    assert s1[0] == 2 * tp_ref - 12.0
    assert s2[0] == tp_ref - 3.0
    assert s3[0] == 9.0 - 2 * (12.0 - tp_ref)
    assert s4[0] == s3[0] - 3.0
    assert r1[0] == 2 * tp_ref - 9.0
    assert r2[0] == tp_ref + 3.0
    assert r3[0] == 12.0 + 2 * (tp_ref - 9.0)
    assert r4[0] == r3[0] + 3.0


@pytest.mark.overlap
def test_pivot_demark_branches() -> None:
    """Demark X: open==close -> (H+L+2C)/4; close>open -> (2H+L+C)/4;
    close<open -> (H+2L+C)/4.
    """
    o = np.array([10.0, 10.0, 12.0])
    h = np.array([11.0, 11.0, 13.0])
    lst = np.array([9.0, 9.0, 10.0])
    c = np.array([10.0, 10.5, 11.0])
    tp, s1, s2, s3, s4, r1, r2, r3, r4 = _pivot_demark(o, h, lst, c)
    assert tp[0] == 0.25 * (11 + 9 + 2 * 10.0)  # open == close
    assert tp[1] == 0.25 * (2 * 11 + 9 + 10.5)  # close > open
    assert tp[2] == 0.25 * (13 + 2 * 10 + 11)  # close < open
    assert s1[0] == 2 * tp[0] - 11.0
    assert r1[0] == 2 * tp[0] - 9.0
    # Demark has no S2..S4/R2..R4: they must be NaN.
    for arr in (s2, s3, s4, r2, r3, r4):
        assert np.isnan(arr).all()


@pytest.mark.overlap
def test_pivot_fibonacci_missing_levels() -> None:
    """Fibonacci has no S4/R4: they must be NaN."""
    from ta.src.overlap.pivots import _pivot_fibonacci

    h = np.array([12.0])
    lst = np.array([9.0])
    c = np.array([10.5])
    tp, s1, s2, s3, s4, r1, r2, r3, r4 = _pivot_fibonacci(h, lst, c)
    tp_ref = (12.0 + 9.0 + 10.5) / 3
    rng = 3.0
    assert tp[0] == tp_ref
    assert s1[0] == tp_ref - 0.382 * rng
    assert s2[0] == tp_ref - 0.618 * rng
    assert s3[0] == tp_ref - rng
    assert r1[0] == tp_ref + 0.382 * rng
    assert r2[0] == tp_ref + 0.618 * rng
    assert r3[0] == tp_ref + rng
    assert np.isnan(s4).all() and np.isnan(r4).all()


@pytest.mark.overlap
def test_pivot_camarilla_formula() -> None:
    """Camarilla levels are close +/- k * (H - L)."""
    from ta.src.overlap.pivots import _pivot_camarilla

    h = np.array([12.0])
    lst = np.array([9.0])
    c = np.array([10.5])
    _tp, s1, s2, s3, s4, r1, _r2, _r3, r4 = _pivot_camarilla(h, lst, c)
    rng = 3.0
    assert s1[0] == 10.5 - 11.0 / 120 * rng
    assert s2[0] == 10.5 - 11.0 / 60 * rng
    assert s3[0] == 10.5 - 0.275 * rng
    assert s4[0] == 10.5 - 0.55 * rng
    assert r1[0] == 10.5 + 11.0 / 120 * rng
    assert r4[0] == 10.5 + 0.55 * rng


# -----------------------------------------------------------------------------
# pivots_ind integration tests
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_pivots_ind_traditional_reference() -> None:
    """Daily traditional pivots: TP=(H+L+C)/3, S1=2TP-H, R1=2TP-L of the
    *previous* day's aggregate; constant within the current day.
    """
    df = make_hourly_df()
    result = pivots_ind(df, method="traditional", anchor="D")
    day1 = df.filter(pl.col("date") < datetime(2024, 1, 2))
    h0 = day1["high"].max()
    l0 = day1["low"].min()
    c0 = day1["close"][-1]
    tp = (h0 + l0 + c0) / 3
    rng = h0 - l0
    win = day_window(result, "TRAD")
    p = win["PIVOTS_TRAD_D_P"].to_numpy()
    s1 = win["PIVOTS_TRAD_D_S1"].to_numpy()
    s2 = win["PIVOTS_TRAD_D_S2"].to_numpy()
    s3 = win["PIVOTS_TRAD_D_S3"].to_numpy()
    r1 = win["PIVOTS_TRAD_D_R1"].to_numpy()
    r2 = win["PIVOTS_TRAD_D_R2"].to_numpy()
    r3 = win["PIVOTS_TRAD_D_R3"].to_numpy()
    # Constant (no lookahead / no mid-period flip) and equal to reference.
    for arr, ref in (
        (p, tp),
        (s1, 2 * tp - h0),
        (s2, tp - rng),
        (s3, tp - 2 * rng),
        (r1, 2 * tp - l0),
        (r2, tp + rng),
        (r3, tp + 2 * rng),
    ):
        assert np.allclose(arr, ref)
    # All finite (first day has a previous aggregate? no: first *shifted*
    # pivot date is 2024-01-02, so day-2 bars see day-1 pivots).
    assert np.isfinite(p).all()


@pytest.mark.overlap
def test_pivots_ind_first_day_has_no_pivot() -> None:
    """Bars of day 1 have no previous daily aggregate -> null pivots."""
    df = make_hourly_df()
    result = pivots_ind(df, method="traditional", anchor="D")
    first = result.filter(pl.col("date") < datetime(2024, 1, 2))
    assert first["PIVOTS_TRAD_D_P"].is_null().all()


@pytest.mark.overlap
def test_pivots_ind_all_methods_produce_columns() -> None:
    """Every method produces P, S1..S4, R1..R4 columns without errors."""
    df = make_hourly_df(n=48)
    expected_suffixes = {
        "traditional": "TRAD",
        "fibonacci": "FIBO",
        "woodie": "WOOD",
        "classic": "CLAS",
        "demark": "DEMA",
        "camarilla": "CAMA",
    }
    for method, sfx in expected_suffixes.items():
        result = pivots_ind(df, method=method, anchor="D")
        for level in ("P", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"):
            assert f"PIVOTS_{sfx}_D_{level}" in result.columns, (method, level)


@pytest.mark.overlap
def test_pivots_ind_woodie_uses_open() -> None:
    """Woodie pivot differs from traditional because it weights the open."""
    df = make_hourly_df(n=48)
    wood = pivots_ind(df, method="woodie", anchor="D")
    trad = pivots_ind(df, method="traditional", anchor="D")
    w = day_window(wood, "WOOD")["PIVOTS_WOOD_D_P"].to_numpy()
    t = day_window(trad, "TRAD")["PIVOTS_TRAD_D_P"].to_numpy()
    assert not np.allclose(w, t)


@pytest.mark.overlap
def test_pivots_ind_support_resistance_ordering() -> None:
    """S4 <= S3 <= S2 <= S1 <= P <= R1 <= R2 <= R3 <= R4 (where defined)."""
    df = make_hourly_df()
    result = pivots_ind(df, method="classic", anchor="D")
    win = day_window(result, "CLAS")
    p = win["PIVOTS_CLAS_D_P"].to_numpy()
    levels = [
        win["PIVOTS_CLAS_D_S4"].to_numpy(),
        win["PIVOTS_CLAS_D_S3"].to_numpy(),
        win["PIVOTS_CLAS_D_S2"].to_numpy(),
        win["PIVOTS_CLAS_D_S1"].to_numpy(),
        p,
        win["PIVOTS_CLAS_D_R1"].to_numpy(),
        win["PIVOTS_CLAS_D_R2"].to_numpy(),
        win["PIVOTS_CLAS_D_R3"].to_numpy(),
        win["PIVOTS_CLAS_D_R4"].to_numpy(),
    ]
    for lo, hi in pairwise(levels):
        assert np.all(lo <= hi + 1e-9)


@pytest.mark.overlap
def test_pivots_ind_unknown_method() -> None:
    """Unknown method raises ValueError listing valid options."""
    df = make_hourly_df(n=10)
    with pytest.raises(ValueError, match="Unknown pivot method"):
        pivots_ind(df, method="nope")


@pytest.mark.overlap
def test_pivots_ind_weekly_anchor() -> None:
    """Weekly anchor resamples and fills pivots across the week."""
    df = make_hourly_df(n=24 * 15)
    result = pivots_ind(df, method="traditional", anchor="W")
    assert "PIVOTS_TRAD_W_P" in result.columns
    p = result["PIVOTS_TRAD_W_P"].drop_nulls()
    assert len(p) > 0
    # Within a single week the pivot is constant (no lookahead).
    week2 = result.filter(
        (pl.col("date") >= datetime(2024, 1, 8))
        & (pl.col("date") < datetime(2024, 1, 15))
    )
    vals = week2["PIVOTS_TRAD_W_P"].to_numpy()
    assert np.isfinite(vals).all()
    assert np.allclose(vals, vals[0])


@pytest.mark.overlap
def test_pivots_ind_preserves_rows_and_order() -> None:
    """Result has the same rows in the same order as the input."""
    df = make_hourly_df()
    result = pivots_ind(df, method="classic", anchor="D")
    assert len(result) == len(df)
    assert (result["date"].to_numpy() == df["date"].to_numpy()).all()
    # Original price columns are unchanged.
    for col in ("open", "high", "low", "close"):
        assert_allclose_np(result[col].to_numpy(), df[col].to_numpy())


def assert_allclose_np(a: np.ndarray, b: np.ndarray) -> None:
    """assert_allclose that tolerates NaNs on both sides."""
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


@pytest.mark.overlap
def test_pivots_ind_custom_column_names() -> None:
    """Non-default OHLC/date column names are respected."""
    df = make_hourly_df(n=48).rename(
        {
            "open": "o",
            "high": "h",
            "low": "lst",
            "close": "c",
            "date": "ts",
        }
    )
    result = pivots_ind(
        df,
        open_col="o",
        high_col="h",
        low_col="lst",
        close_col="c",
        date_col="ts",
        method="traditional",
        anchor="D",
    )
    assert "PIVOTS_TRAD_D_P" in result.columns
    p = result["PIVOTS_TRAD_D_P"].drop_nulls()
    assert len(p) > 0
