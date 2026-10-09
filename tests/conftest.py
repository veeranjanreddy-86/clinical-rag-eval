"""Shared fixtures: a tiny synthetic corpus and an offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from clinical_rag.config import Settings
from clinical_rag.generate import ExtractiveGenerator
from clinical_rag.index import BM25Retriever
from clinical_rag.ingest import Chunk, chunk_documents, load_documents

if TYPE_CHECKING:
    from clinical_rag.pipeline import RAGPipeline

DOCS = {
    "imaging-policy": """# Imaging Policy

> SYNTHETIC test document.

## Validity

An approved imaging authorization is valid for 60 calendar days from the approval date.

## Timeframes

Standard imaging requests receive a decision within 3 business days.
""",
    "falls-protocol": """# Falls Protocol

> SYNTHETIC test document.

## Risk

A fall risk score of 45 or higher identifies a patient as high fall risk.

## After a Fall

Notify the provider within 15 minutes after any fall.
""",
    "interpreter-policy": """# Interpreter Policy

> SYNTHETIC test document.

## Interpreters

Qualified medical interpreters are available by video 24 hours a day at no cost.
""",
}


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / f"{name}.md").write_text(text, encoding="utf-8")
    return d


@pytest.fixture
def settings(docs_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        docs_dir=docs_dir,
        chunks_path=tmp_path / "missing" / "chunks.jsonl",
        report_path=tmp_path / "report.md",
        chunk_size=40,
        chunk_overlap=10,
        top_k=3,
        min_score=1.0,
        min_query_coverage=0.4,
    )


@pytest.fixture
def chunks(settings: Settings) -> list[Chunk]:
    docs = load_documents(settings.docs_dir)
    return chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)


@pytest.fixture
def pipeline(chunks: list[Chunk], settings: Settings) -> RAGPipeline:
    from clinical_rag.pipeline import RAGPipeline

    return RAGPipeline(BM25Retriever(chunks), ExtractiveGenerator(), settings)
