"""FastAPI service exposing ``POST /ask`` and ``GET /health``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from clinical_rag import __version__
from clinical_rag.config import get_settings
from clinical_rag.logging_utils import configure_logging
from clinical_rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    doc_id: str
    chunk_id: str
    title: str


class RetrievalOut(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    rank: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]
    retrieval: list[RetrievalOut]
    redactions: dict[str, int]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    retriever: str
    generator: str


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """Create the app. Pass a pipeline to inject one (tests); otherwise built from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level)
        app.state.pipeline = pipeline or RAGPipeline.from_settings(settings)
        logger.info("service ready", extra={"retriever": app.state.pipeline.retriever.name})
        yield

    app = FastAPI(title="Clinical RAG", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        p: RAGPipeline = request.app.state.pipeline
        return HealthResponse(
            status="ok", version=__version__, retriever=p.retriever.name, generator=p.generator.name
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        p: RAGPipeline = request.app.state.pipeline
        try:
            result = p.ask(body.question, top_k=body.top_k)
        except Exception as exc:
            logger.exception("ask failed")
            raise HTTPException(status_code=502, detail="Answer generation failed") from exc
        return AskResponse(
            answer=result.answer,
            refused=result.refused,
            citations=[CitationOut(**c.__dict__) for c in result.citations],
            retrieval=[
                RetrievalOut(
                    chunk_id=r.chunk.chunk_id, doc_id=r.chunk.doc_id, score=r.score, rank=r.rank
                )
                for r in result.retrieved
            ],
            redactions=result.redactions,
            latency_ms=result.latency_ms,
        )

    return app


app = create_app()
