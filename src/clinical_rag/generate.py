"""Answer generation.

:class:`ExtractiveGenerator` is the default: deterministic, offline, and every sentence it emits
is copied verbatim from a retrieved chunk with an inline ``[doc_id]`` citation. LLM providers
(OpenAI / Azure OpenAI) sit behind the same :class:`Generator` interface and are only constructed
when explicitly configured.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from clinical_rag.config import Settings
from clinical_rag.index import ScoredChunk
from clinical_rag.text import split_sentences, tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Citation:
    """A pointer from an answer to the chunk that supports it."""

    doc_id: str
    chunk_id: str
    title: str


@dataclass(frozen=True)
class Answer:
    """Generated answer text with its citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)


class Generator(Protocol):
    """Composes an answer from a question and supporting chunks."""

    name: str

    def generate(self, question: str, results: Sequence[ScoredChunk]) -> Answer: ...


def _citation(result: ScoredChunk) -> Citation:
    c = result.chunk
    return Citation(doc_id=c.doc_id, chunk_id=c.chunk_id, title=c.title)


class ExtractiveGenerator:
    """Select the query-relevant sentences from the top chunks and cite each one."""

    name = "offline-extractive"

    def __init__(self, max_sentences: int = 3, relative_cutoff: float = 0.5) -> None:
        self.max_sentences = max_sentences
        self.relative_cutoff = relative_cutoff

    def generate(self, question: str, results: Sequence[ScoredChunk]) -> Answer:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, str, ScoredChunk]] = []
        for result in results:
            for pos, sentence in enumerate(split_sentences(result.chunk.text)):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap:
                    # Prefer high term overlap, then higher-ranked chunks, then earlier sentences.
                    score = overlap + 0.1 / result.rank
                    candidates.append((score, pos, sentence, result))
        if not candidates:
            return Answer(text="", citations=[])
        best = max(c[0] for c in candidates)
        candidates = [c for c in candidates if c[0] >= best * self.relative_cutoff]
        candidates.sort(key=lambda c: (-c[0], c[3].rank, c[1]))

        chosen: list[tuple[str, ScoredChunk]] = []
        seen: set[str] = set()
        for _, _, sentence, result in candidates:
            if sentence not in seen:
                seen.add(sentence)
                chosen.append((sentence, result))
            if len(chosen) == self.max_sentences:
                break

        text = " ".join(f"{s.rstrip()} [{r.chunk.doc_id}]" for s, r in chosen)
        citations: dict[str, Citation] = {}
        for _, r in chosen:
            citations.setdefault(r.chunk.chunk_id, _citation(r))
        return Answer(text=text, citations=list(citations.values()))


_SYSTEM_PROMPT = (
    "You answer questions for care teams using ONLY the provided context passages. "
    "Cite every claim with the passage id in square brackets, e.g. [doc-id#0]. "
    "If the context does not contain the answer, reply exactly: I don't know."
)
_CITE = re.compile(r"\[([a-z0-9][a-z0-9\-]*(?:#\d+)?)\]")


class ChatCompletionGenerator:
    """LLM generator for any OpenAI-compatible chat client (OpenAI or Azure OpenAI)."""

    def __init__(self, client: Any, model: str, name: str) -> None:
        self._client = client
        self._model = model
        self.name = name

    def generate(self, question: str, results: Sequence[ScoredChunk]) -> Answer:
        context = "\n\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in results)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        by_chunk = {r.chunk.chunk_id: r for r in results}
        by_doc = {r.chunk.doc_id: r for r in reversed(results)}
        citations: dict[str, Citation] = {}
        for ref in _CITE.findall(text):
            hit = by_chunk.get(ref) or by_doc.get(ref.split("#")[0])
            if hit:  # ignore citations to passages that were not provided
                citations.setdefault(hit.chunk.chunk_id, _citation(hit))
        return Answer(text=text, citations=list(citations.values()))


def build_generator(settings: Settings) -> Generator:
    """Construct the configured generator; LLM providers require their env vars."""
    if settings.generator == "offline":
        return ExtractiveGenerator(max_sentences=settings.max_answer_sentences)
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("Install the 'llm' extra: pip install .[llm]") from exc
    if settings.generator == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("CRAG_GENERATOR=openai requires OPENAI_API_KEY")
        client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
        return ChatCompletionGenerator(client, settings.llm_model, "openai")
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        raise RuntimeError(
            "CRAG_GENERATOR=azure_openai requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
    client = openai.AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
    )
    model = settings.azure_openai_deployment or settings.llm_model
    return ChatCompletionGenerator(client, model, "azure_openai")
