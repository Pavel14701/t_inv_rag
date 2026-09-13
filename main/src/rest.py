"""REST layer over WhiteAPI (TZ-10, endpoint map p.2.2).

aiohttp-based minimal HTTP: no new dependencies (aiohttp ships with
aiogram). All request/response bodies use msgspec structs from
``contracts/`` -- no local duplicates.
"""

from __future__ import annotations

import traceback

from collections.abc import Awaitable, Callable
from typing import Any

import msgspec

from aiohttp import web
from msgspec import json as msgspec_json

from contracts import BacktestCommand, OhlcvBatch


JSON = web.Response
Routes = Callable[[web.Request], Awaitable[web.Response]]

API_KEY: web.AppKey[Any] = web.AppKey("api", object)


def _json(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _decode_error(exc: Exception) -> web.Response:
    return _json({"error": str(exc)}, status=400)


def build_app(
    api: Any,
    strategies_provider: Callable[[], list[dict[str, Any]]] | None = None,
    strategy_detail: Callable[[str], dict[str, Any] | None] | None = None,
    rag_generate: Callable[[str], dict[str, Any]] | None = None,
) -> web.Application:
    """Build the white API REST application."""
    app = web.Application()
    app[API_KEY] = api

    async def ingest_candles(request: web.Request) -> web.Response:
        """POST /ingest/candles: idempotent candle ingest."""
        try:
            body = await request.read()
            batch = msgspec_json.decode(body, type=OhlcvBatch)
        except msgspec.DecodeError as exc:
            return _decode_error(exc)
        n = request.app[API_KEY].candles.ingest_batch(batch)
        return _json({"ingested": n})

    async def list_strategies(request: web.Request) -> web.Response:  # noqa: RUF029
        """GET /strategies: read-only registry."""
        if strategies_provider is None:
            return _json([], status=200)
        return _json(strategies_provider())

    async def get_strategy(request: web.Request) -> web.Response:  # noqa: RUF029
        """GET /strategies/{id}: DSL text, AST, metrics."""
        sid = request.match_info["sid"]
        if strategy_detail is None:
            return _json({"error": "not found"}, status=404)
        detail = strategy_detail(sid)
        if detail is None:
            return _json({"error": "not found"}, status=404)
        return _json(detail)

    async def submit_backtest(request: web.Request) -> web.Response:
        """POST /backtests: 202 + job_id (async via queue)."""
        try:
            body = await request.read()
            cmd = msgspec_json.decode(body, type=BacktestCommand)
        except msgspec.DecodeError as exc:
            return _decode_error(exc)
        job = request.app[API_KEY].submit_backtest(cmd)
        return _json({"job_id": job.job_id}, status=202)

    async def backtest_status(request: web.Request) -> web.Response:  # noqa: RUF029
        """GET /backtests/{job_id}: status/report from evt.report."""
        job_id = request.match_info["job_id"]
        status = request.app[API_KEY].get_backtest_status(job_id)
        if status is None:
            return _json({"error": "not found"}, status=404)
        return _json(status)

    async def get_signals(request: web.Request) -> web.Response:  # noqa: RUF029
        """GET /signals?ticker=...: latest signals with P(win)."""
        ticker = request.query.get("ticker", "")
        if not ticker:
            return _json({"error": "ticker required"}, status=400)
        limit = int(request.query.get("limit", "10"))
        return _json(request.app[API_KEY].get_signals(ticker, limit))

    async def handle_rag_generate(request: web.Request) -> web.Response:
        """POST /rag/generate: RAG DSL generation (local contour)."""
        if rag_generate is None:
            return _json({"error": "rag contour not available"}, status=503)
        try:
            body = await request.read()
            data = msgspec_json.decode(body)
        except msgspec.DecodeError as exc:
            return _decode_error(exc)
        task = data.get("task", "")
        if not task:
            return _json({"error": "task required"}, status=400)
        return _json(rag_generate(task))

    app.router.add_post("/ingest/candles", ingest_candles)
    app.router.add_get("/strategies", list_strategies)
    app.router.add_get("/strategies/{sid}", get_strategy)
    app.router.add_post("/backtests", submit_backtest)
    app.router.add_get("/backtests/{job_id}", backtest_status)
    app.router.add_get("/signals", get_signals)
    app.router.add_post("/rag/generate", handle_rag_generate)
    return app


def error_middleware() -> Any:
    """Middleware converting unhandled errors to 500 JSON."""

    @web.middleware
    async def mw(request: web.Request, handler: Routes) -> web.Response:
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception:
            traceback.print_exc()
            return _json({"error": "internal"}, status=500)

    return mw
