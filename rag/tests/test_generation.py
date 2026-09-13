"""Tests for RAG generation: prompt, repair loop, mocked LLM (TZ-07 п.3)."""

from __future__ import annotations

from rag.generation import generate_dsl
from rag.llm import LLMRouter


def _mock_router(responses: list[str]) -> LLMRouter:
    """Create a router with a scripted sequence of responses."""
    call_count = 0
    router = LLMRouter(default_provider="mock")

    class MockProvider:
        name = "mock"

        def complete(self, prompt, options):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        async def acomplete(self, prompt, options):
            return self.complete(prompt, options)

    router.register("mock", MockProvider())
    return router


MANIFEST = {
    "indicators": {
        "rsi": {"attributes": ["value"], "parameters": {}},
        "close": {"attributes": []},
        "open": {"attributes": []},
        "ema": {"attributes": ["value"]},
        "rising": {},
    }
}


class TestGenerateDSL:
    def test_valid_first_attempt(self) -> None:
        """LLM returns valid DSL on first try -> status ok, 1 iteration."""
        router = _mock_router(["rsi.value < 30"])
        result = generate_dsl(
            router,
            "RSI oversold",
            MANIFEST,
        )
        assert result.status == "ok"
        assert result.dsl_entry == "rsi.value < 30"
        assert result.iterations == 1

    def test_repair_on_invalid_indicator(self) -> None:
        """First attempt uses unknown indicator, repair fixes it."""
        router = _mock_router(
            [
                "stoch_rsi < 20",  # invalid indicator
                "rsi.value < 20",  # repaired
            ]
        )
        result = generate_dsl(router, "RSI oversold", MANIFEST)
        assert result.status == "ok"
        assert result.iterations == 2
        assert result.dsl_entry == "rsi.value < 20"

    def test_repair_exhausted(self) -> None:
        """LLM keeps producing invalid DSL -> status failed after max."""
        router = _mock_router(
            [
                "bad_indicator < 1",
                "another_bad > 2",
                "still_bad < 3",
            ]
        )
        result = generate_dsl(
            router, "test", MANIFEST, max_repair_iterations=2
        )
        assert result.status == "failed"
        assert result.iterations == 3  # 1 initial + 2 repairs

    def test_markdown_fences_stripped(self) -> None:
        """LLM wraps DSL in markdown fences -> stripped."""
        router = _mock_router(
            [
                "```dsl\nrsi.value < 30\n```",
            ]
        )
        result = generate_dsl(router, "RSI oversold", MANIFEST)
        assert result.status == "ok"
        assert result.dsl_entry == "rsi.value < 30"

    def test_docs_included_in_result(self) -> None:
        router = _mock_router(["rsi.value < 30"])
        docs = [{"heading": "RSI Guide", "text": "RSI measures momentum"}]
        result = generate_dsl(router, "RSI", MANIFEST, docs=docs)
        assert "RSI Guide" in result.docs_used

    def test_prompt_contains_manifest(self) -> None:
        """The prompt must contain the manifest (deterministic context)."""
        captured_prompts = []
        router = _mock_router(["rsi.value < 30"])
        original = router.get("mock").complete

        def capturing_complete(prompt, options):
            captured_prompts.append(prompt)
            return original(prompt, options)

        router.get("mock").complete = capturing_complete
        generate_dsl(router, "test task", MANIFEST)
        assert len(captured_prompts) == 1
        assert "rsi" in captured_prompts[0]
        assert "test task" in captured_prompts[0]
