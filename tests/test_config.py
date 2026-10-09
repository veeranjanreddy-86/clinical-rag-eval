import pytest

from clinical_rag.config import Settings
from clinical_rag.generate import ExtractiveGenerator, build_generator


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_TOP_K", "7")
    assert Settings().top_k == 7


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_size=20, chunk_overlap=20)


def test_default_generator_is_offline() -> None:
    assert isinstance(build_generator(Settings()), ExtractiveGenerator)


def test_llm_generator_requires_credentials_or_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_generator(Settings(generator="openai"))
