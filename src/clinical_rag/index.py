"""Pluggable retrievers.

The default :class:`BM25Retriever` is pure Python with no external dependencies. The optional
:class:`EmbeddingRetriever` accepts any :class:`Embedder`; concrete embedders import their
client libraries lazily, so they are never required for tests or the offline path.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from clinical_rag.config import Settings
from clinical_rag.ingest import Chunk
from clinical_rag.text import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    """Anything that can rank chunks for a query."""

    name: str

    def search(self, query: str, k: int) -> list[ScoredChunk]: ...


class BM25Retriever:
    """Okapi BM25 over chunk search text."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(c.search_text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = sum(self._lens) / len(self._lens)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        """Return the BM25 score of every chunk for ``query``."""
        terms = tokenize(query)
        scores = []
        for tf, dl in zip(self._tfs, self._lens, strict=True):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                    s += self._idf[t] * f * (self.k1 + 1) / norm
            scores.append(s)
        return scores

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            ScoredChunk(self.chunks[i], round(scores[i], 4), r)
            for r, i in enumerate(order, start=1)
            if scores[i] > 0
        ]


class Embedder(Protocol):
    """Maps texts to dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'embeddings' extra: pip install .[embeddings]") from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - optional
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


class OpenAIEmbedder:
    """Embeddings via the OpenAI API; only usable when ``OPENAI_API_KEY`` is configured."""

    def __init__(self, settings: Settings, model: str = "text-embedding-3-small") -> None:
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extra: pip install .[llm]") from exc
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingRetriever:
    """Dense retriever using cosine similarity (brute force; fine for small corpora)."""

    name = "embeddings"

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("EmbeddingRetriever requires at least one chunk")
        self.chunks = chunks
        self._embedder = embedder
        self._vectors = embedder.embed([c.search_text for c in chunks])

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        q = self._embedder.embed([query])[0]
        scores = [_cosine(q, v) for v in self._vectors]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [ScoredChunk(self.chunks[i], round(scores[i], 4), r) for r, i in enumerate(order, 1)]


def build_retriever(chunks: list[Chunk], settings: Settings) -> Retriever:
    """Construct the configured retriever."""
    if settings.retriever == "embeddings":
        embedder: Embedder
        if settings.embedding_backend == "openai":
            embedder = OpenAIEmbedder(settings)
        else:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
        logger.info("using embedding retriever", extra={"backend": settings.embedding_backend})
        return EmbeddingRetriever(chunks, embedder)
    return BM25Retriever(chunks)
