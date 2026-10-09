from pathlib import Path

import pytest

from clinical_rag.ingest import (
    chunk_text,
    load_chunks,
    load_documents,
    save_chunks,
    split_sections,
)


def test_chunk_text_overlap_and_coverage() -> None:
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), chunk_size=10, overlap=3)
    assert [len(c.split()) for c in chunks] == [10, 10, 10, 4]
    # consecutive chunks share exactly `overlap` words
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    # every word is covered and the last chunk ends at the last word
    assert set(" ".join(chunks).split()) == set(words)
    assert chunks[-1].split()[-1] == "w24"


def test_chunk_text_short_and_empty() -> None:
    assert chunk_text("one two three", 10, 2) == ["one two three"]
    assert chunk_text("   ", 10, 2) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (5, 5), (5, -1)])
def test_chunk_text_rejects_bad_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size, overlap)


def test_split_sections_drops_banner() -> None:
    sections = split_sections("# Title\n\n> SYNTHETIC banner\n\nIntro.\n\n## Part\n\nBody text.")
    assert sections == [("Title", "Intro."), ("Part", "Body text.")]


def test_documents_and_chunk_roundtrip(docs_dir: Path, chunks, tmp_path: Path) -> None:
    docs = load_documents(docs_dir)
    assert {d.doc_id for d in docs} == {"imaging-policy", "falls-protocol", "interpreter-policy"}
    assert all("SYNTHETIC" not in c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks


def test_load_documents_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
