import logging
from dataclasses import dataclass
from typing import Any

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any]


class PineconeRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index = None
        self._embeddings = None

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector = self._embedding_client().embed_query(query)
        result = self._pinecone_index().query(
            vector=vector,
            top_k=self.settings.retrieval_top_k,
            namespace=self.settings.pinecone_namespace,
            include_metadata=True,
        )
        matches = getattr(result, "matches", None)
        if matches is None and isinstance(result, dict):
            matches = result.get("matches", [])
        matches = matches or []
        chunks: list[RetrievedChunk] = []
        for match in matches:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
            metadata = dict(metadata or {})
            text = str(metadata.pop("text", "")).strip()
            if not text:
                continue
            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score", 0.0)
            chunks.append(RetrievedChunk(text=text, score=score, metadata=metadata))
        logger.info("Retrieved %s chunks for query", len(chunks))
        return chunks

    def _embedding_client(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _pinecone_index(self):
        if self._index is None:
            from pinecone import Pinecone

            client = Pinecone(api_key=self.settings.pinecone_api_key)
            if self.settings.pinecone_host:
                self._index = client.Index(host=self.settings.pinecone_host)
            else:
                self._index = client.Index(self.settings.pinecone_index_name)
        return self._index


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No company information was found."
    return "\n\n".join(chunk.text for chunk in chunks)
