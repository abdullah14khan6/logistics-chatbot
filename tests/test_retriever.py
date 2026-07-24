from backend.config.settings import Settings
from backend.rag.retriever import PineconeRetriever, RetrievedChunk, format_context


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, query: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2]


class FakeIndex:
    def __init__(self) -> None:
        self.calls = 0
        self.options = {}

    def query(self, **kwargs):
        self.calls += 1
        self.options = kwargs
        return {
            "matches": [
                {
                    "score": 0.9,
                    "metadata": {
                        "text": "Warehousing is available.",
                        "document_name": "company.pdf",
                        "page_number": 20,
                        "section": "Warehousing",
                    },
                }
            ]
        }


def test_retriever_caches_queries_and_filters_staff_content() -> None:
    retriever = PineconeRetriever(Settings(retrieval_cache_size=4))
    embeddings = FakeEmbeddings()
    index = FakeIndex()
    retriever._embeddings = embeddings
    retriever._index = index

    first = retriever.retrieve("warehousing", {"staff_directory"})
    second = retriever.retrieve("warehousing", {"staff_directory"})

    assert first == second
    assert embeddings.calls == 1
    assert index.calls == 1
    assert index.options["filter"] == {
        "content_type": {"$nin": ["staff_directory"]}
    }


def test_formatted_context_keeps_internal_evidence_metadata() -> None:
    context = format_context(
        [
            RetrievedChunk(
                text="Warehousing is available.",
                score=0.9,
                metadata={
                    "document_name": "company.pdf",
                    "page_number": 20,
                    "section": "Warehousing",
                },
            )
        ]
    )

    assert "<evidence_item" in context
    assert "page=20" in context
    assert "Warehousing is available." in context
