import json
from pathlib import Path

import pytest

from clinical_rag.evaluate import (
    citation_precision,
    groundedness,
    load_gold,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
)


def test_metric_functions() -> None:
    assert recall_at_k(["a", "b", "c"], ["c", "d"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert reciprocal_rank(["x", "a", "b"], ["b", "a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    assert citation_precision(["a", "a", "b"], ["a"]) == 0.5
    assert citation_precision([], ["a"]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


def test_groundedness_heuristic() -> None:
    chunk = "Approved authorization is valid for 60 days."
    assert groundedness("Authorization is valid for 60 days. [doc]", [chunk]) == 1.0
    assert groundedness("Pizza is served on Fridays.", [chunk]) == 0.0
    assert groundedness("", [chunk]) == 0.0


def test_run_evaluation_on_tiny_fixture(pipeline, settings, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    rows = [
        {
            "id": "1",
            "question": "How long is an imaging authorization valid?",
            "answerable": True,
            "gold_doc_ids": ["imaging-policy"],
        },
        {
            "id": "2",
            "question": "When must the provider be notified after a fall?",
            "answerable": True,
            "gold_doc_ids": ["falls-protocol"],
        },
        {
            "id": "3",
            "question": "What is the parking fee for visitors?",
            "answerable": False,
            "gold_doc_ids": [],
        },
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    summary = run_evaluation(pipeline, gold, settings.report_path)
    assert summary["questions"] == 3
    assert summary["recall@3"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["citation_precision"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    report = settings.report_path.read_text(encoding="utf-8")
    assert "# Evaluation Report" in report and "| 3 | False | True |" in report


def test_load_gold_rejects_bad_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_gold(bad)
