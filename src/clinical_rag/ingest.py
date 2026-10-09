"""Load markdown documents and split them into overlapping, section-aware chunks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Document:
    """A source document. ``doc_id`` is the file stem and is used in citations."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def search_text(self) -> str:
        """Text used for indexing: section heading plus body."""
        return f"{self.section}. {self.text}"


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every ``*.md`` file in ``docs_dir`` (sorted for determinism)."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    docs: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next(
            (m.group(2).strip() for line in text.splitlines() if (m := _HEADING.match(line))),
            path.stem,
        )
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    if not docs:
        raise ValueError(f"No markdown documents found in {docs_dir}")
    logger.info("loaded documents", extra={"count": len(docs), "docs_dir": str(docs_dir)})
    return docs


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs.

    Blockquote lines (used for the synthetic-data banner) are dropped from bodies so that
    boilerplate does not dominate retrieval.
    """
    sections: list[tuple[str, str]] = []
    heading, lines = "", []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            if any(s.strip() for s in lines):
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = match.group(2).strip(), []
        elif not line.lstrip().startswith(">"):
            lines.append(line)
    if any(s.strip() for s in lines):
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of ``chunk_size`` words that overlap by ``overlap`` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each document section-by-section; chunk ids are ``<doc_id>#<n>``."""
    chunks: list[Chunk] = []
    for doc in docs:
        n = 0
        for section, body in split_sections(doc.text):
            for piece in chunk_text(body, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{n}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=section or doc.title,
                        text=piece,
                    )
                )
                n += 1
    logger.info("chunked documents", extra={"documents": len(docs), "chunks": len(chunks)})
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Persist chunks as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk)) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Load chunks previously written by :func:`save_chunks`."""
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def ingest(docs_dir: Path, chunks_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Load, chunk and persist the corpus. Returns the chunks."""
    chunks = chunk_documents(load_documents(docs_dir), chunk_size, overlap)
    save_chunks(chunks, chunks_path)
    return chunks
