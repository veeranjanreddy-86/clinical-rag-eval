"""Environment-driven configuration (pydantic-settings).

All settings use the ``CRAG_`` prefix (e.g. ``CRAG_TOP_K=5``). Provider credentials use their
conventional, unprefixed names (``OPENAI_API_KEY``, ``AZURE_OPENAI_*``) and are optional.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for ingestion, retrieval, generation and evaluation."""

    model_config = SettingsConfigDict(env_prefix="CRAG_", env_file=".env", extra="ignore")

    docs_dir: Path = Path("data/docs")
    chunks_path: Path = Path("artifacts/chunks.jsonl")
    eval_path: Path = Path("data/eval/gold.jsonl")
    report_path: Path = Path("reports/eval_report.md")

    chunk_size: int = Field(default=80, ge=10, description="Chunk size in words.")
    chunk_overlap: int = Field(default=20, ge=0, description="Overlap between chunks in words.")
    top_k: int = Field(default=4, ge=1, le=20)

    retriever: Literal["bm25", "embeddings"] = "bm25"
    embedding_backend: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    generator: Literal["offline", "openai", "azure_openai"] = "offline"
    llm_model: str = "gpt-4o-mini"

    # Refusal guardrail: the top chunk must clear BOTH thresholds to be treated as support.
    min_score: float = Field(default=4.0, ge=0.0, description="Minimum top retrieval score.")
    min_query_coverage: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Min fraction of query terms found in top chunk."
    )
    max_answer_sentences: int = Field(default=3, ge=1)

    log_level: str = "INFO"

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "CRAG_OPENAI_API_KEY")
    )
    azure_openai_endpoint: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_API_KEY")
    )
    azure_openai_deployment: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT")
    )
    azure_openai_api_version: str = Field(
        default="2024-06-01", validation_alias=AliasChoices("AZURE_OPENAI_API_VERSION")
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
