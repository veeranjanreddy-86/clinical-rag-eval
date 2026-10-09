"""Shared text utilities: tokenization, light stemming and sentence splitting."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# Sentence ends, plus markdown bullets (which survive chunking as " - " after ':' / ';' / '.').
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.:;])\s+[-*]\s+|\n\s*[-*]\s+|\n{2,}")

STOPWORDS = frozenset(
    """a about after all also an and any are as at be been before being by can could do does
    for from had has have how i if in into is it its may must no not of on or our should so such
    than that the their them then there these they this those to under up was we were what when
    where which while who whom why will with within would you your""".split()
)


def stem(token: str) -> str:
    """Very light suffix stripping so that e.g. 'authorizations' matches 'authorization'."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, and stem."""
    tokens = _TOKEN.findall(text.lower())
    return [stem(t) for t in tokens if keep_stopwords or t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences / bullet items, stripping list markers and whitespace."""
    parts = _SENTENCE.split(text.strip())
    out = []
    for part in parts:
        cleaned = re.sub(r"^\s*[-*]\s+", "", part).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            out.append(cleaned)
    return out
