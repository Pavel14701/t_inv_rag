"""Data loaders for the inference script (TZ-05 п.2.3).

Sources: synthetic (tests/dev), parquet (tests), yfinance (dev fallback),
T-Invest (production; lazy import, requires ``INVEST_TOKEN``).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from .provider import _COLUMN_ALIASES


_DATE_CANDIDATES = ('date', 'time', 'datetime', 'timestamp')


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a frame to unified OHLCV + date names.

    Accepts both the legacy ``PriceDataFramePolars`` schema
    (``open_price`` etc.) and the unified TZ-02 one.

    """
    rename: dict[str, str] = {}
    for canon, aliases in _COLUMN_ALIASES.items():
        if canon in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canon
                break
    if rename:
        df = df.rename(rename)
    if 'date' not in df.columns:
        for cand in _DATE_CANDIDATES:
            if cand in df.columns:
                df = df.rename({cand: 'date'})
                break
        else:
            raise ValueError(
                'no date column found; expected one of '
                f'{_DATE_CANDIDATES}'
            )
    return df


def load_synthetic(n_bars: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Synthetic GBM series with trend regimes (for tests)."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        np.full(n_bars // 3, 0.0005),
        np.full(n_bars // 3, -0.0004),
        np.full(n_bars - 2 * (n_bars // 3), 0.0002),
    ])
    ret = drift + rng.normal(0, 0.01, n_bars)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 100_000, n_bars).astype(np.int64)
    dates = pl.Series(
        'date',
        [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_bars)],
    ).cast(pl.Datetime('us'))
    return pl.DataFrame({
        'date': dates[:n_bars],
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


def load_parquet(path: str) -> pl.DataFrame:
    """Load a frame from parquet with name normalization."""
    return normalize(pl.read_parquet(path))


def load_yfinance(ticker: str, period_from: str, period_to: str):
    """Dev fallback via yfinance (daily candles)."""
    import yfinance as yf

    raw = yf.download(ticker, start=period_from, end=period_to,
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f'yfinance returned no data for {ticker!r}')
    if isinstance(raw.columns, __import__(
        'pandas'
    ).MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pl.from_pandas(raw.reset_index())
    df = df.rename({
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume',
    })
    return normalize(df)


def load_tinvest(ticker: str, period_from: str, period_to: str,
                 interval: str = '1d'):
    """Primary source: T-Invest (t_tech.invest). Requires INVEST_TOKEN.

    Lazy import: the trading API package is not needed for dev runs.

    """
    import os

    from t_tech.invest import Client

    token = os.environ.get('INVEST_TOKEN')
    if not token:
        raise RuntimeError('INVEST_TOKEN is not set')
    from datetime import datetime

    start = datetime.fromisoformat(period_from)
    end = datetime.fromisoformat(period_to)
    with Client(token) as client:
        instrument = client.instruments.share_by(
            id_type=2, class_code='TQBR', ticker=ticker
        ).instrument
        candles = client.market_data.get_candles(
            instrument_id=instrument.uid,
            interval=interval,
            from_=start,
            to=end,
        )
    rows = [
        {
            'date': c.time,
            'open': float(c.open),
            'high': float(c.high),
            'low': float(c.low),
            'close': float(c.close),
            'volume': int(c.volume),
        }
        for c in candles.candles
    ]
    if not rows:
        raise ValueError(f'T-Invest returned no candles for {ticker!r}')
    return pl.DataFrame(rows)


def load(source: str, **kwargs) -> pl.DataFrame:
    """Source dispatcher."""
    if source == 'synthetic':
        return load_synthetic(**kwargs)
    if source == 'parquet':
        return load_parquet(kwargs['path'])
    if source == 'yfinance':
        return load_yfinance(
            kwargs['ticker'], kwargs['period_from'], kwargs['period_to']
        )
    if source == 'tinvest':
        return load_tinvest(
            kwargs['ticker'], kwargs['period_from'], kwargs['period_to']
        )
    raise ValueError(f'unknown source {source!r}')