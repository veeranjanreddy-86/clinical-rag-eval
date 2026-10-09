from itertools import pairwise

import pytest

from clinical_rag.index import BM25Retriever
from clinical_rag.text import tokenize


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("The authorizations are VALID") == ["authorization", "valid"]


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("How long is an imaging authorization valid?", "imaging-policy"),
        ("What fall risk score means high fall risk?", "falls-protocol"),
        ("Is video interpreting available at night?", "interpreter-policy"),
    ],
)
def test_bm25_ranks_relevant_doc_first(chunks, query: str, expected_doc: str) -> None:
    results = BM25Retriever(chunks).search(query, k=3)
    assert results[0].chunk.doc_id == expected_doc
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(a.score >= b.score for a, b in pairwise(results))


def test_bm25_no_match_returns_empty(chunks) -> None:
    assert BM25Retriever(chunks).search("zebra xylophone", k=3) == []


def test_bm25_requires_chunks() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([])
