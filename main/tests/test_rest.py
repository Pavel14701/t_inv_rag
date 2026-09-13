"""Tests for the white API REST layer (TZ-10)."""

from __future__ import annotations

import pytest

from aiohttp.test_utils import TestClient, TestServer
from msgspec import json as msgspec_json

from contracts import BacktestCommand, Candle, OhlcvBatch
from main.src.api import WhiteAPI
from main.src.rest import build_app


def _batch() -> OhlcvBatch:
    return OhlcvBatch(
        inst_id="BTC-USDT",
        candles=[
            Candle(
                inst_id="BTC-USDT",
                ts=1000 + i,
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=10.0,
            )
            for i in range(3)
        ],
    )


@pytest.fixture
def client_factory():
    def make(api: WhiteAPI, **kw) -> TestClient:
        app = build_app(api, **kw)
        return TestClient(TestServer(app))

    return make


@pytest.mark.asyncio
async def test_ingest_candles_idempotent(client_factory) -> None:
    client = client_factory(WhiteAPI())
    async with client:
        body = msgspec_json.encode(_batch())
        resp = await client.post("/ingest/candles", data=body)
        assert resp.status == 200
        assert (await resp.json())["ingested"] == 3
        resp = await client.post("/ingest/candles", data=body)
        assert (await resp.json())["ingested"] == 0  # duplicates skipped


@pytest.mark.asyncio
async def test_ingest_invalid_body_400(client_factory) -> None:
    client = client_factory(WhiteAPI())
    async with client:
        resp = await client.post("/ingest/candles", data=b"{bad json")
        assert resp.status == 400


@pytest.mark.asyncio
async def test_backtest_202_flow(client_factory) -> None:
    """POST /backtests -> 202 + job_id; report arrives via handle_report."""
    api = WhiteAPI()
    client = client_factory(api)
    async with client:
        cmd = BacktestCommand(
            request_id="job-9", strategy_id="s1", dsl_entry="rsi.value < 30"
        )
        resp = await client.post("/backtests", data=msgspec_json.encode(cmd))
        assert resp.status == 202
        job_id = (await resp.json())["job_id"]

        # simulate local node finishing the job
        from main.src.bridge import report_from_payload

        api.handle_report(
            msgspec_json.encode(
                report_from_payload(job_id, "completed", {"pf": 1.4})
            )
        )
        resp = await client.get(f"/backtests/{job_id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "completed"
        assert data["report"]["pf"] == 1.4


@pytest.mark.asyncio
async def test_backtest_unknown_job_404(client_factory) -> None:
    client = client_factory(WhiteAPI())
    async with client:
        resp = await client.get("/backtests/nope")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_signals_endpoint(client_factory) -> None:
    from contracts import SignalEvent

    api = WhiteAPI()
    api.handle_signal(
        msgspec_json.encode(
            SignalEvent(
                inst_id="BTC-USDT",
                ts=1,
                direction="long",
                entry_price=1.0,
                sl_price=0.9,
                tp_price=1.2,
                p_win=0.7,
            )
        )
    )
    client = client_factory(api)
    async with client:
        resp = await client.get("/signals?ticker=BTC-USDT")
        assert resp.status == 200
        data = await resp.json()
        assert data[0]["p_win"] == 0.7
        resp = await client.get("/signals")
        assert resp.status == 400


@pytest.mark.asyncio
async def test_strategies_endpoints(client_factory) -> None:
    client = client_factory(
        WhiteAPI(),
        strategies_provider=lambda: [{"id": "s1"}],
        strategy_detail=lambda sid: (
            {"id": sid, "dsl": "rsi.value < 30"} if sid == "s1" else None
        ),
    )
    async with client:
        resp = await client.get("/strategies")
        assert await resp.json() == [{"id": "s1"}]
        resp = await client.get("/strategies/s1")
        assert (await resp.json())["dsl"] == "rsi.value < 30"
        resp = await client.get("/strategies/none")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_rag_generate_503_and_ok(client_factory) -> None:
    client = client_factory(WhiteAPI())
    async with client:
        resp = await client.post(
            "/rag/generate", data=msgspec_json.encode({"task": "t"})
        )
        assert resp.status == 503

    client = client_factory(
        WhiteAPI(), rag_generate=lambda task: {"dsl_entry": "rsi < 30"}
    )
    async with client:
        resp = await client.post(
            "/rag/generate", data=msgspec_json.encode({"task": "t"})
        )
        assert resp.status == 200
        assert (await resp.json())["dsl_entry"] == "rsi < 30"
