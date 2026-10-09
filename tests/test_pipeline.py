from dataclasses import replace

from clinical_rag.generate import ExtractiveGenerator
from clinical_rag.guardrails import REFUSAL_TEXT
from clinical_rag.index import BM25Retriever
from clinical_rag.pipeline import RAGPipeline


def test_answer_has_citations_from_retrieved_chunks(pipeline) -> None:
    result = pipeline.ask("How long is an imaging authorization valid?")
    assert not result.refused
    assert "60 calendar days" in result.answer
    assert "[imaging-policy]" in result.answer
    retrieved_ids = {r.chunk.chunk_id for r in result.retrieved}
    assert result.citations
    assert {c.chunk_id for c in result.citations} <= retrieved_ids


def test_unanswerable_question_is_refused_without_citations(pipeline) -> None:
    result = pipeline.ask("How many vacation days do employees get?")
    assert result.refused
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_phi_in_question_is_redacted_and_still_answered(pipeline) -> None:
    question = "MRN 7654321, call 555-222-3333: how long is imaging authorization valid?"
    result = pipeline.ask(question)
    assert result.redactions == {"MRN": 1, "PHONE": 1}
    assert not result.refused
    assert "7654321" not in result.answer


def test_deterministic(pipeline) -> None:
    q = "When must the provider be notified after a fall?"
    assert pipeline.ask(q).answer == pipeline.ask(q).answer


def test_phi_in_source_text_is_redacted_from_output(chunks, settings) -> None:
    leaky = [
        replace(c, text=c.text + " Contact intake@example.org for imaging authorization.")
        if c.doc_id == "imaging-policy"
        else c
        for c in chunks
    ]
    pipe = RAGPipeline(BM25Retriever(leaky), ExtractiveGenerator(max_sentences=5), settings)
    result = pipe.ask("Who to contact for imaging authorization?")
    assert "intake@example.org" not in result.answer
    assert result.redactions.get("output_EMAIL", 0) >= 1
