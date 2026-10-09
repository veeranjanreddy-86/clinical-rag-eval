import pytest

from clinical_rag.guardrails import (
    REFUSAL_TEXT,
    assess_support,
    redact,
    strip_placeholders,
)
from clinical_rag.index import BM25Retriever


@pytest.mark.parametrize(
    ("text", "kind", "secret"),
    [
        ("SSN is 123-45-6789.", "SSN", "123-45-6789"),
        ("Email jane.doe@example.org today", "EMAIL", "jane.doe@example.org"),
        ("Call (555) 123-4567 now", "PHONE", "123-4567"),
        ("Call +1 555.123.4567 now", "PHONE", "555.123.4567"),
        ("Patient MRN: 00123456 admitted", "MRN", "00123456"),
        ("medical record number AB-1234567", "MRN", "1234567"),
        ("my MRN is 12345678", "MRN", "12345678"),
        ("DOB 03/14/1961", "DOB", "03/14/1961"),
        ("date of birth: 1961-03-14", "DOB", "1961-03-14"),
        ("born on March 14, 1961", "DOB", "1961"),
    ],
)
def test_redact_patterns(text: str, kind: str, secret: str) -> None:
    result = redact(text)
    assert secret not in result.text
    assert f"[REDACTED_{kind}]" in result.text
    assert result.counts.get(kind) == 1


def test_redact_leaves_policy_text_alone() -> None:
    text = "Decision within 3 business days; dial extension 4-5555; valid 60 days."
    result = redact(text)
    assert result.text == text
    assert result.total == 0


def test_strip_placeholders_removes_labels() -> None:
    redacted = redact("Is MRN 1234567 eligible? Email a@b.co").text
    assert strip_placeholders(redacted) == "Is eligible? Email"


def test_refusal_when_no_supporting_chunk(chunks) -> None:
    retriever = BM25Retriever(chunks)
    query = "What is the parking fee for visitors?"
    decision = assess_support(query, retriever.search(query, 3), 1.0, 0.4)
    assert not decision.supported
    assert REFUSAL_TEXT.startswith("I don't know")


def test_support_when_evidence_strong(chunks) -> None:
    retriever = BM25Retriever(chunks)
    query = "How long is an imaging authorization valid?"
    decision = assess_support(query, retriever.search(query, 3), 1.0, 0.4)
    assert decision.supported and decision.reason == "ok"


def test_low_coverage_refuses_even_with_high_score(chunks) -> None:
    retriever = BM25Retriever(chunks)
    query = "imaging vendor contract pricing warranty maintenance"
    results = retriever.search(query, 3)
    assert results  # 'imaging' matches...
    decision = assess_support(query, results, 0.1, 0.4)
    assert not decision.supported and decision.reason == "low_coverage"
