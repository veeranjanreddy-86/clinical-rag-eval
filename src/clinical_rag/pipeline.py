"""End-to-end RAG pipeline: redact -> retrieve -> check support -> generate -> redact."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from clinical_rag.config import Settings
from clinical_rag.generate import Answer, Citation, Generator, build_generator
from clinical_rag.guardrails import (
    REFUSAL_TEXT,
    SupportDecision,
    assess_support,
    redact,
    strip_placeholders,
)
from clinical_rag.index import Retriever, ScoredChunk, build_retriever
from clinical_rag.ingest import Chunk, chunk_documents, load_chunks, load_documents

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskResult:
    """Everything the API / CLI / evaluator needs about one question."""

    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    refused: bool
    support: SupportDecision
    redactions: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class RAGPipeline:
    """Composable pipeline; retriever and generator are injected for testability."""

    def __init__(self, retriever: Retriever, generator: Generator, settings: Settings) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        """Build from persisted chunks if present, otherwise chunk the docs directory."""
        chunks: list[Chunk]
        if settings.chunks_path.exists():
            chunks = load_chunks(settings.chunks_path)
        else:
            docs = load_documents(settings.docs_dir)
            chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        return cls(build_retriever(chunks, settings), build_generator(settings), settings)

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        """Answer ``question`` with citations, or refuse when evidence is insufficient."""
        start = time.perf_counter()
        k = top_k or self.settings.top_k
        red_in = redact(question)
        query = strip_placeholders(red_in.text)

        retrieved = self.retriever.search(query, k) if query else []
        support = assess_support(
            query, retrieved, self.settings.min_score, self.settings.min_query_coverage
        )

        answer = Answer(text="")
        if support.supported:
            try:
                answer = self.generator.generate(query, retrieved)
            except Exception:
                logger.exception("generation failed", extra={"generator": self.generator.name})
                raise
        refused = not support.supported or not answer.text or not answer.citations
        refusal_reason = None
        if refused:
            refusal_reason = support.reason if not support.supported else "no_answer"
            answer = Answer(text=REFUSAL_TEXT, citations=[])

        red_out = redact(answer.text)
        redactions = dict(red_in.counts)
        for kind, n in red_out.counts.items():
            redactions[f"output_{kind}"] = n
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "ask",
            extra={
                "retriever": self.retriever.name,
                "generator": self.generator.name,
                "top_score": support.top_score,
                "coverage": support.coverage,
                "refused": refused,
                "refusal_reason": refusal_reason,
                "citations": [c.chunk_id for c in answer.citations],
                "redactions": redactions,
                "latency_ms": latency_ms,
            },
        )
        return AskResult(
            answer=red_out.text,
            citations=answer.citations,
            retrieved=retrieved,
            refused=refused,
            support=support,
            redactions=redactions,
            latency_ms=latency_ms,
        )
