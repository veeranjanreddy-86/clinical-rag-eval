"""Evaluation harness over a JSONL gold set; writes a markdown report.

Metrics
-------
- recall@k: fraction of gold documents found in the top-k retrieved chunks (answerable only).
- MRR: reciprocal rank of the first retrieved chunk from a gold document (answerable only).
- citation precision: fraction of cited documents that are gold documents (answered only).
- groundedness: mean share of answer-sentence tokens found in the cited chunks (answered only).
- refusal accuracy: share of unanswerable questions correctly refused.
- false refusal rate: share of answerable questions wrongly refused.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from clinical_rag.pipeline import AskResult, RAGPipeline
from clinical_rag.text import split_sentences, tokenize

logger = logging.getLogger(__name__)

_CITE_MARK = re.compile(r"\s*\[[^\]]+\]")


@dataclass(frozen=True)
class GoldItem:
    """One evaluation question."""

    id: str
    question: str
    answerable: bool
    gold_doc_ids: list[str]


@dataclass(frozen=True)
class ItemResult:
    """Per-question metrics."""

    item: GoldItem
    retrieved_doc_ids: list[str]
    cited_doc_ids: list[str]
    refused: bool
    recall: float | None
    reciprocal_rank: float | None
    citation_precision: float | None
    groundedness: float | None
    top_score: float
    latency_ms: float


def load_gold(path: Path) -> list[GoldItem]:
    """Load and validate the gold set."""
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                items.append(
                    GoldItem(
                        id=str(raw["id"]),
                        question=str(raw["question"]),
                        answerable=bool(raw["answerable"]),
                        gold_doc_ids=list(raw.get("gold_doc_ids", [])),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid gold record on line {n} of {path}: {exc}") from exc
    return items


def recall_at_k(retrieved_doc_ids: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold docs appearing in the first ``k`` retrieved results."""
    if not gold:
        raise ValueError("recall@k is undefined without gold documents")
    return len(set(retrieved_doc_ids[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first gold document, or 0 when none is retrieved."""
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_doc_ids: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of distinct cited documents that are gold documents."""
    cited = set(cited_doc_ids)
    if not cited:
        return 0.0
    return len(cited & set(gold)) / len(cited)


def groundedness(answer: str, cited_texts: Sequence[str]) -> float:
    """Heuristic: mean over answer sentences of the share of their tokens present in cited text."""
    support = set()
    for text in cited_texts:
        support.update(tokenize(text))
    scores = []
    for sentence in split_sentences(_CITE_MARK.sub("", answer)):
        tokens = tokenize(sentence)
        if tokens:
            scores.append(sum(t in support for t in tokens) / len(tokens))
    return sum(scores) / len(scores) if scores else 0.0


def score_item(item: GoldItem, result: AskResult, k: int) -> ItemResult:
    """Compute metrics for a single question."""
    retrieved = [r.chunk.doc_id for r in result.retrieved]
    cited = [c.doc_id for c in result.citations]
    recall = rr = cprec = ground = None
    if item.answerable:
        recall = recall_at_k(retrieved, item.gold_doc_ids, k)
        rr = reciprocal_rank(retrieved, item.gold_doc_ids)
    if not result.refused:
        cited_ids = {c.chunk_id for c in result.citations}
        texts = [r.chunk.text for r in result.retrieved if r.chunk.chunk_id in cited_ids]
        ground = groundedness(result.answer, texts)
        cprec = citation_precision(cited, item.gold_doc_ids)
    return ItemResult(
        item=item,
        retrieved_doc_ids=retrieved,
        cited_doc_ids=cited,
        refused=result.refused,
        recall=recall,
        reciprocal_rank=rr,
        citation_precision=cprec,
        groundedness=ground,
        top_score=result.support.top_score,
        latency_ms=result.latency_ms,
    )


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(results: Sequence[ItemResult], k: int) -> dict[str, float | int | None]:
    """Aggregate per-question results into headline metrics."""
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]
    answered = [r for r in results if not r.refused]
    latencies = sorted(r.latency_ms for r in results)
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        f"recall@{k}": _mean([r.recall for r in answerable]),
        "mrr": _mean([r.reciprocal_rank for r in answerable]),
        "citation_precision": _mean([r.citation_precision for r in answered]),
        "groundedness": _mean([r.groundedness for r in answered]),
        "refusal_accuracy": (
            sum(r.refused for r in unanswerable) / len(unanswerable) if unanswerable else None
        ),
        "false_refusal_rate": (
            sum(r.refused for r in answerable) / len(answerable) if answerable else None
        ),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_report(
    summary: dict[str, float | int | None], results: Sequence[ItemResult], meta: dict[str, str]
) -> str:
    """Render the evaluation as markdown."""
    lines = ["# Evaluation Report", "", "Synthetic gold set; see `data/eval/gold.jsonl`.", ""]
    lines += ["## Configuration", "", "| Setting | Value |", "|---|---|"]
    lines += [f"| {k} | `{v}` |" for k, v in meta.items()]
    lines += ["", "## Summary", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()]
    lines += [
        "",
        "## Per-question results",
        "",
        "| id | answerable | refused | top score | recall | RR | cite prec | grounded | cited |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cited = ", ".join(dict.fromkeys(r.cited_doc_ids)) or "-"
        lines.append(
            f"| {r.item.id} | {r.item.answerable} | {r.refused} | {r.top_score:.2f} | "
            f"{_fmt(r.recall)} | {_fmt(r.reciprocal_rank)} | {_fmt(r.citation_precision)} | "
            f"{_fmt(r.groundedness)} | {cited} |"
        )
    return "\n".join(lines) + "\n"


def run_evaluation(
    pipeline: RAGPipeline, gold_path: Path, report_path: Path | None = None
) -> dict[str, float | int | None]:
    """Run every gold question through ``pipeline``, write the report, return the summary."""
    k = pipeline.settings.top_k
    items = load_gold(gold_path)
    results = [score_item(item, pipeline.ask(item.question), k) for item in items]
    summary = summarize(results, k)
    if report_path is not None:
        s = pipeline.settings
        meta = {
            "retriever": pipeline.retriever.name,
            "generator": pipeline.generator.name,
            "top_k": str(k),
            "chunk_size / overlap (words)": f"{s.chunk_size} / {s.chunk_overlap}",
            "min_score / min_query_coverage": f"{s.min_score} / {s.min_query_coverage}",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(summary, results, meta), encoding="utf-8")
        logger.info("wrote evaluation report", extra={"path": str(report_path)})
    return summary
