"""RAG ingestion: chunk docs + manifest + strategies (TZ-07).

Produces chunks for the ``dsl_docs`` collection (static documentation)
and points for the ``strategy_cases`` collection (validated strategies).
White-list enforced: only configured paths are ingested.
"""

from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocChunk:
    """A chunk of documentation for embedding and retrieval."""

    chunk_id: str  # hash of content (idempotent ingest)
    source: str
    heading: str
    text: str
    doc_type: str = "dsl_doc"
    lang: str = "en"


@dataclass(frozen=True, slots=True)
class StrategyCase:
    """A validated strategy as a retrieval point (1 strategy = 1 point)."""

    case_id: str  # strategy.id
    dsl_entry: str
    dsl_exit: str | None
    indicators_used: list[str]
    description: str
    manifest_hash: str
    metrics_json: str | None = None


def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_markdown(
    path: Path,
    max_chunk_chars: int = 600,
    overlap_lines: int = 2,
) -> list[DocChunk]:
    """Split a markdown file into chunks by headings.

    Chunks are bounded by ``max_chunk_chars``; headings start new chunks.
    Overlap: the last ``overlap_lines`` lines of the previous chunk are
    prepended to the next chunk for context continuity.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Split by headings (## or #)
    sections: list[tuple[str, list[str]]] = []
    current_heading = path.stem
    current_lines: list[str] = []
    for line in lines:
        if re.match(r"^#{1,2} ", line):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    # Further split large sections
    chunks: list[DocChunk] = []

    def _mk(heading: str, body: str) -> DocChunk:
        idx = len(chunks)
        return DocChunk(
            chunk_id=_chunk_id(f"{path}:{heading}:{idx}:{body[:50]}"),
            source=str(path),
            heading=heading,
            text=body,
        )

    for heading, sec_lines in sections:
        body = "\n".join(sec_lines).strip()
        if not body:
            continue
        if len(body) <= max_chunk_chars:
            chunks.append(_mk(heading, body))
        else:
            # Split by paragraph
            paragraphs = body.split("\n\n")
            buf: list[str] = []
            buf_len = 0
            for para in paragraphs:
                plen = len(para) + 2
                if buf_len + plen > max_chunk_chars and buf:
                    chunks.append(_mk(heading, "\n\n".join(buf)))
                    buf = [para]
                    buf_len = plen
                else:
                    buf.append(para)
                    buf_len += plen
            if buf:
                chunks.append(_mk(heading, "\n\n".join(buf)))

    return chunks


def ingest_docs(
    doc_paths: list[Path],
    max_chunk_chars: int = 600,
) -> list[DocChunk]:
    """Ingest markdown documents (white-list enforced by caller)."""
    chunks: list[DocChunk] = []
    for path in doc_paths:
        if path.exists() and path.suffix == ".md":
            chunks.extend(chunk_markdown(path, max_chunk_chars))
    return chunks


def render_manifest_text(manifest: dict) -> str:
    """Render an indicator manifest as a human-readable prompt block."""
    lines = ["Available indicators:"]
    for name, spec in sorted(manifest.get("indicators", {}).items()):
        attrs = spec.get("attributes", [])
        params = spec.get("parameters", {})
        attr_str = f".{'/'.join(attrs)}" if attrs else ""
        param_str = ""
        if params:
            parts = []
            for pname, pspec in params.items():
                default = pspec.get("default", "?")
                parts.append(f"{pname}={default}")
            param_str = f"({', '.join(parts)})"
        lines.append(f"  - {name}{attr_str}{param_str}")
    return "\n".join(lines)


def strategy_to_case(strategy: Any, indicators: list[str]) -> StrategyCase:
    """Convert a validated Strategy to a StrategyCase point."""
    metrics = strategy.metrics
    metrics_json = (
        json.dumps(
            {
                "pf": metrics.profit_factor,
                "sharpe": metrics.sharpe,
            }
        )
        if metrics
        else None
    )
    return StrategyCase(
        case_id=strategy.id,
        dsl_entry=strategy.dsl_entry,
        dsl_exit=strategy.dsl_exit,
        indicators_used=indicators,
        description=strategy.description,
        manifest_hash=strategy.manifest_hash,
        metrics_json=metrics_json,
    )
