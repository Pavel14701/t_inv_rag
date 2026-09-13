"""Tests for RAG ingestion: chunking, manifest rendering (TZ-07)."""

from __future__ import annotations

import pathlib

import pytest

from rag.ingestion import (
    chunk_markdown,
    ingest_docs,
    render_manifest_text,
    strategy_to_case,
)


@pytest.fixture
def sample_md(tmp_path: pathlib.Path) -> pathlib.Path:
    md = tmp_path / "test_doc.md"
    md.write_text(
        "# RSI Guide\n\nThe Relative Strength Index measures momentum.\n"
        "Values below 30 indicate oversold conditions.\n\n"
        "## Usage in DSL\n\nUse `rsi.value < 30` for oversold entry.\n\n"
        "## Confirmation\n\nCombine with `rising(close, 5)` for trend.\n",
        encoding="utf-8",
    )
    return md


class TestChunking:
    def test_creates_chunks_by_heading(self, sample_md: pathlib.Path) -> None:
        chunks = chunk_markdown(sample_md)
        assert len(chunks) >= 3  # RSI Guide, Usage, Confirmation
        assert all(c.text for c in chunks)

    def test_chunk_ids_unique(self, sample_md: pathlib.Path) -> None:
        chunks = chunk_markdown(sample_md)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # idempotent

    def test_headings_extracted(self, sample_md: pathlib.Path) -> None:
        chunks = chunk_markdown(sample_md)
        headings = [c.heading for c in chunks]
        assert any("RSI Guide" in h for h in headings)

    def test_large_doc_split(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "large.md"
        md.write_text(
            "# Big\n\n"
            + "\n\n".join(f"Paragraph {i} " + "x" * 100 for i in range(20)),
            encoding="utf-8",
        )
        chunks = chunk_markdown(md, max_chunk_chars=300)
        assert len(chunks) > 1

    def test_ingest_docs_white_list(self, sample_md: pathlib.Path) -> None:
        chunks = ingest_docs([sample_md])
        assert len(chunks) > 0
        # Non-existent path → no chunks (no error)
        assert ingest_docs([pathlib.Path("/nonexistent.md")]) == []


class TestManifestRendering:
    def test_renders_indicators(self) -> None:
        manifest = {
            "indicators": {
                "rsi": {"attributes": ["value"], "parameters": {}},
                "ema": {
                    "attributes": ["value"],
                    "parameters": {"length": {"type": "int", "default": 20}},
                },
            }
        }
        text = render_manifest_text(manifest)
        assert "rsi" in text
        assert "ema" in text
        assert "value" in text
        assert "length=20" in text

    def test_empty_manifest(self) -> None:
        text = render_manifest_text({})
        assert "Available indicators" in text


class TestStrategyToCase:
    def test_converts_strategy(self) -> None:
        from strategies.src.application.strategy import Strategy

        s = Strategy(
            id="s1",
            name="test",
            description="test strategy",
            dsl_entry="rsi.value < 30",
            manifest_hash="abc123",
        )
        case = strategy_to_case(s, ["rsi"])
        assert case.case_id == "s1"
        assert case.dsl_entry == "rsi.value < 30"
        assert "rsi" in case.indicators_used
        assert case.manifest_hash == "abc123"
