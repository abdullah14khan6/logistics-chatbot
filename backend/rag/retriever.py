import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
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
        self._client_lock = RLock()
        self._cache_lock = RLock()
        self._cache: OrderedDict[
            tuple[str, tuple[str, ...]], list[RetrievedChunk]
        ] = OrderedDict()

    def warmup(self) -> None:
        embeddings = self._embedding_client()
        embeddings.embed_query("logistics services")
        self._pinecone_index()

    def retrieve(
        self,
        query: str,
        exclude_content_types: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        exclusions = tuple(sorted(exclude_content_types or set()))
        cache_key = (" ".join(query.lower().split()), exclusions)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return list(cached)

        vector = self._embedding_client().embed_query(query)
        query_options: dict[str, Any] = {
            "vector": vector,
            "top_k": self.settings.retrieval_candidate_k,
            "namespace": self.settings.pinecone_namespace,
            "include_metadata": True,
        }
        if exclusions:
            query_options["filter"] = {
                "content_type": {"$nin": list(exclusions)}
            }
        result = self._pinecone_index().query(
            **query_options,
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
        chunks = self._rerank(query, chunks)[: self.settings.retrieval_top_k]
        logger.info("Retrieved %s chunks for query", len(chunks))
        with self._cache_lock:
            self._cache[cache_key] = list(chunks)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.settings.retrieval_cache_size:
                self._cache.popitem(last=False)
        return chunks

    @staticmethod
    def _rerank(
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2
        }

        def rank_key(chunk: RetrievedChunk) -> tuple[float, float]:
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
            overlap = (
                len(query_terms.intersection(chunk_terms)) / len(query_terms)
                if query_terms
                else 0.0
            )
            return chunk.score + (0.08 * overlap), overlap

        return sorted(chunks, key=rank_key, reverse=True)

    def _embedding_client(self):
        if self._embeddings is None:
            with self._client_lock:
                if self._embeddings is not None:
                    return self._embeddings
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model_name,
                    model_kwargs={
                        "local_files_only": self.settings.embedding_local_files_only
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
        return self._embeddings

    def _pinecone_index(self):
        if self._index is None:
            with self._client_lock:
                if self._index is not None:
                    return self._index
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
    items = []
    for index, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("document_name", "company knowledge"))
        page = chunk.metadata.get("page_number", "")
        section = str(chunk.metadata.get("section", "")).strip()
        attributes = [f"source={source!r}"]
        if page:
            attributes.append(f"page={page!r}")
        if section:
            attributes.append(f"section={section!r}")
        items.append(
            f"<evidence_item id={index!r} {' '.join(attributes)}>\n"
            f"{chunk.text}\n"
            "</evidence_item>"
        )
    return "\n\n".join(items)
