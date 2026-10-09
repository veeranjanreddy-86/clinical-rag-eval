"""Guardrails: PHI/PII redaction and evidence-based refusal.

Redaction is regex-based and intentionally conservative; it is a safety net, not a certified
de-identification method (see README "Limitations").
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from clinical_rag.index import ScoredChunk
from clinical_rag.text import tokenize

REFUSAL_TEXT = (
    "I don't know. The indexed documents do not contain enough supporting evidence "
    "to answer this question."
)

_DATE = (
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})"
)

# Order matters: more specific patterns run first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "DOB",
        re.compile(
            rf"\b(?P<label>DOB|D\.O\.B\.?|date of birth|born(?: on)?)\s*[:\-]?\s*{_DATE}",
            re.IGNORECASE,
        ),
    ),
    (
        "MRN",
        re.compile(
            r"\b(?P<label>MRN|medical record (?:number|no\.?|#)|patient id)\s*(?:is\s+)?[:#]?\s*"
            r"[A-Z]{0,3}-?\d{5,12}\b",
            re.IGNORECASE,
        ),
    ),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b")),
]

_PLACEHOLDER = re.compile(
    r"(?:\b(?:MRN|DOB|D\.O\.B\.?|date of birth|born(?: on)?|medical record (?:number|no\.?|#)"
    r"|patient id)\s+)?\[REDACTED_[A-Z]+\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text plus a count of redactions per entity type."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Replace SSNs, emails, phone numbers, MRN-like identifiers and DOBs with placeholders."""
    counts: Counter[str] = Counter()
    for kind, pattern in _PATTERNS:
        placeholder = f"[REDACTED_{kind}]"

        def _sub(match: re.Match[str], kind: str = kind, placeholder: str = placeholder) -> str:
            counts[kind] += 1
            label = match.groupdict().get("label")
            return f"{label} {placeholder}" if label else placeholder

        text = pattern.sub(_sub, text)
    return RedactionResult(text=text, counts=dict(counts))


def strip_placeholders(text: str) -> str:
    """Remove redaction placeholders so they do not influence retrieval."""
    return re.sub(r"\s{2,}", " ", _PLACEHOLDER.sub(" ", text)).strip()


@dataclass(frozen=True)
class SupportDecision:
    """Whether retrieved evidence is strong enough to attempt an answer."""

    supported: bool
    top_score: float
    coverage: float
    reason: str


def query_coverage(query: str, chunk_text: str) -> float:
    """Fraction of distinct query terms that appear in ``chunk_text``."""
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(chunk_text))) / len(terms)


def assess_support(
    query: str, results: Sequence[ScoredChunk], min_score: float, min_coverage: float
) -> SupportDecision:
    """Decide whether to answer.

    A result counts as support only if its retrieval score is at least ``min_score`` AND it
    covers at least ``min_coverage`` of the query terms. The second check catches queries that
    share a single high-IDF word with the corpus but are otherwise off-topic.
    """
    if not results:
        return SupportDecision(False, 0.0, 0.0, "no_results")
    top_score = results[0].score
    best_cov = max(query_coverage(query, r.chunk.search_text) for r in results)
    qualifying = [
        r
        for r in results
        if r.score >= min_score and query_coverage(query, r.chunk.search_text) >= min_coverage
    ]
    if qualifying:
        return SupportDecision(True, top_score, round(best_cov, 3), "ok")
    reason = "low_score" if top_score < min_score else "low_coverage"
    return SupportDecision(False, top_score, round(best_cov, 3), reason)
