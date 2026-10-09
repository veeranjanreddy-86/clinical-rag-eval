"""Provider interfaces exercised with in-process fakes (no network, no optional deps)."""

from types import SimpleNamespace

from clinical_rag.generate import ChatCompletionGenerator
from clinical_rag.index import BM25Retriever, EmbeddingRetriever


class FakeChatClient:
    def __init__(self, reply: str) -> None:
        self.calls: list[dict] = []
        create = self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self._reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self._reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_chat_generator_parses_only_known_citations(chunks) -> None:
    results = BM25Retriever(chunks).search("imaging authorization valid", 2)
    top = results[0].chunk
    client = FakeChatClient(f"Valid for 60 days [{top.chunk_id}] and [unknown-doc#3].")
    answer = ChatCompletionGenerator(client, "test-model", "fake").generate("q", results)
    assert [c.chunk_id for c in answer.citations] == [top.chunk_id]
    sent = client.calls[0]
    assert sent["temperature"] == 0 and top.chunk_id in sent["messages"][1]["content"]


class BagOfWordsEmbedder:
    """Deterministic toy embedder: counts of a fixed vocabulary."""

    vocab = ("imaging", "authorization", "fall", "risk", "interpreter", "video")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(t.lower().count(w)) for w in self.vocab] for t in texts]


def test_embedding_retriever_with_fake_embedder(chunks) -> None:
    retriever = EmbeddingRetriever(chunks, BagOfWordsEmbedder())
    results = retriever.search("fall risk", k=2)
    assert results[0].chunk.doc_id == "falls-protocol"
    assert retriever.name == "embeddings"
