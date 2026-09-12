r"""CLI скрипта инференса (TZ-05 п.3.1).

Пример:
    python -m infer.cli --source synthetic --entry "close > ema(length=20)"
    python -m infer.cli --source parquet --parquet data.parquet \
        --strategy-file strategy.json --output out.csv

Формат strategy-file (заготовка формата Strategy TZ-02):
    {"dsl_entry": "...", "dsl_exit": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys

import polars as pl

from .data import load
from .engine import run_inference


def build_parser() -> argparse.ArgumentParser:
    """CLI-парсер."""
    p = argparse.ArgumentParser(
        prog='infer',
        description='DSL strategy inference over OHLCV data (TZ-05).',
    )
    p.add_argument(
        '--source',
        choices=['synthetic', 'parquet', 'yfinance', 'tinvest'],
        default='synthetic',
    )
    p.add_argument('--parquet', help='path for --source parquet')
    p.add_argument('--ticker', help='ticker for yfinance/tinvest')
    p.add_argument('--from', dest='period_from', help='ISO date from')
    p.add_argument('--to', dest='period_to', help='ISO date to')
    p.add_argument('--entry', help='DSL entry expression')
    p.add_argument('--exit', dest='dsl_exit', help='DSL exit expression')
    p.add_argument(
        '--strategy-file',
        help='JSON file {"dsl_entry": ..., "dsl_exit": ...}',
    )
    p.add_argument('--ml', help='path to model bundle (TZ-06)')
    p.add_argument('--p-threshold', type=float, default=None)
    p.add_argument('--output', default='signals.csv')
    p.add_argument('--bars', type=int, default=1000,
                   help='bars for synthetic source')
    return p


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    args = build_parser().parse_args(argv)

    dsl_entry, dsl_exit = _resolve_strategy(args)
    if not dsl_entry:
        print('error: --entry or --strategy-file is required',
              file=sys.stderr)
        return 2

    kwargs = {}
    if args.source == 'parquet':
        if not args.parquet:
            print('error: --parquet is required', file=sys.stderr)
            return 2
        kwargs = {'path': args.parquet}
    elif args.source in ('yfinance', 'tinvest'):
        kwargs = {
            'ticker': args.ticker,
            'period_from': args.period_from,
            'period_to': args.period_to,
        }
    elif args.source == 'synthetic':
        kwargs = {'n_bars': args.bars}

    df = load(args.source, **kwargs)

    predictor = None
    if args.ml:
        from ai.src.bundle import EntryExitPredictor, load_bundle

        predictor = EntryExitPredictor(load_bundle(args.ml))
        if args.p_threshold is None:
            print('error: --ml requires --p-threshold', file=sys.stderr)
            return 2

    result = run_inference(
        df, dsl_entry, dsl_exit,
        p_threshold=args.p_threshold, predictor=predictor,
    )
    _write_output(result.signals, args.output)
    print(result.summary())
    if result.errors:
        print(f'{len(result.errors)} provider errors (first 5):')
        for line in result.errors[:5]:
            print(f'  {line}')
    return 0


def _resolve_strategy(args) -> tuple[str | None, str | None]:
    """Взять DSL из --strategy-file или из --entry/--exit."""
    if args.strategy_file:
        with open(args.strategy_file, encoding='utf-8') as fh:
            data = json.load(fh)
        return data.get('dsl_entry'), data.get('dsl_exit')
    return args.entry, args.dsl_exit


def _write_output(signals: pl.DataFrame, output: str) -> None:
    if output.endswith('.parquet'):
        signals.write_parquet(output)
    else:
        signals.write_csv(output)


if __name__ == '__main__':
    raise SystemExit(main())