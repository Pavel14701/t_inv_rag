"""RAG generation: prompt template + LLM call + repair loop (TZ-07 п.3).

Prompt includes: rendered manifest (deterministic), retrieved docs
(semantic context), and few-shot examples (validated strategies).
Output is machine-validated: parse → indicator check → repair ≤ 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsl.exceptions import ParseError
from dsl.parser import parse
from rag.llm import LLMRouter


GENERATION_PROMPT = """\
You are a DSL generator for a deterministic trading system.

## Available indicators (ONLY use these - never invent indicators):
{manifest_text}

## DSL syntax:
- Comparison: `indicator.value < 30`, `close > open`
- Logic: `and`, `or`, `not`
- Functions: `rising(close, 5)`, `falling(close, 3)`

## Reference documentation:
{docs_text}

## Example strategies:
{examples_text}

## Task:
{task}

Generate a DSL expression for the entry condition. Return ONLY the
expression, no explanations, no markdown formatting."""


REPAIR_PROMPT = """\
Your previous DSL expression had errors:
{errors}

Previous expression: {previous_dsl}

Fix the expression and return ONLY the corrected DSL expression."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of a RAG generation attempt."""

    status: str  # "ok" | "failed"
    dsl_entry: str
    dsl_exit: str | None = None
    errors: list[str] = field(default_factory=list)
    iterations: int = 0
    docs_used: list[str] = field(default_factory=list)


def _validate_dsl(dsl: str, manifest: dict) -> list[str]:
    """Validate a DSL expression: parse + indicator check."""
    try:
        tree = parse(dsl)
    except ParseError as exc:
        return [f"DSL parse error: {exc}"]
    known = set(manifest.get("indicators") or {})
    found = _find_indicators(tree.to_dict())
    return [
        f"Unknown indicator '{name}'; available: {sorted(known)}"
        for name in found
        if name not in known
    ]


def _find_indicators(node: Any) -> set[str]:
    """Recursively extract indicator names from a to_dict tree."""
    result: set[str] = set()
    if isinstance(node, dict):
        if node.get("type") == "IndicatorAccess" and node.get("indicator"):
            result.add(node["indicator"])
        for v in node.values():
            result.update(_find_indicators(v))
    elif isinstance(node, list):
        for item in node:
            result.update(_find_indicators(item))
    return result


def _render_manifest(manifest: dict) -> str:
    """Render an indicator manifest as a human-readable prompt block."""
    lines = ["Available indicators:"]
    for name, spec in sorted(manifest.get("indicators", {}).items()):
        attrs = spec.get("attributes", [])
        params = spec.get("parameters", {})
        attr_str = f".{'/'.join(attrs)}" if attrs else ""
        param_str = ""
        if params:
            pparts = [
                f"{pname}={pspec.get('default', '?')}"
                for pname, pspec in params.items()
            ]
            param_str = f"({', '.join(pparts)})"
        lines.append(f"  - {name}{attr_str}{param_str}")
    return "\n".join(lines)


def _format_docs(docs: list[dict[str, str]]) -> str:
    if not docs:
        return "(no documentation retrieved)"
    return "\n\n".join(
        f"### {d.get('heading', 'Doc')}\n{d.get('text', '')}" for d in docs
    )


def _format_examples(examples: list[dict[str, str]]) -> str:
    if not examples:
        return "(no examples)"
    return "\n\n".join(
        f"Strategy: {e.get('description', '')}\nDSL: {e.get('dsl_entry', '')}"
        for e in examples
    )


def generate_dsl(
    router: LLMRouter,
    task: str,
    manifest: dict,
    docs: list[dict[str, str]] | None = None,
    examples: list[dict[str, str]] | None = None,
    max_repair_iterations: int = 2,
    temperature: float = 0.3,
) -> GenerationResult:
    """Generate a DSL expression from a natural language task.

    Uses RAG: renders manifest + docs + examples into the prompt,
    calls the LLM, validates the output, and repairs if needed (max 2).

    Returns:
        GenerationResult with status, DSL, errors, and iteration count.

    """
    from rag.ingestion import render_manifest_text

    manifest_text = render_manifest_text(manifest)
    docs_text = _format_docs(docs or [])
    examples_text = _format_examples(examples or [])

    prompt = GENERATION_PROMPT.format(
        manifest_text=manifest_text,
        docs_text=docs_text,
        examples_text=examples_text,
        task=task,
    )
    from rag.llm import CompletionOptions

    options = CompletionOptions(temperature=temperature)

    errors: list[str] = []
    dsl_entry = ""
    iterations = 0

    for attempt in range(max_repair_iterations + 1):
        iterations = attempt + 1
        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = REPAIR_PROMPT.format(
                errors="\n".join(errors),
                previous_dsl=dsl_entry,
                indicator_names=manifest_text,
            )
        dsl_entry = router.complete(current_prompt, options=options).strip()
        if dsl_entry.startswith("```"):
            dsl_entry = "\n".join(
                line
                for line in dsl_entry.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        errors = _validate_dsl(dsl_entry, manifest)
        if not errors:
            return GenerationResult(
                status="ok",
                dsl_entry=dsl_entry,
                errors=[],
                iterations=iterations,
                docs_used=[d.get("heading", "") for d in (docs or [])],
            )

    return GenerationResult(
        status="failed",
        dsl_entry=dsl_entry,
        errors=errors,
        iterations=iterations,
    )
