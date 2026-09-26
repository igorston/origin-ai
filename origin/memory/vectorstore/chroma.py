from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

from origin.config import Settings

MetadataValue = str | int | float | bool


class MemoryHit(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, MetadataValue]


class VectorMemory:
    """Long-term semantic memory backed by a local Chroma collection (cosine similarity)."""

    def __init__(self, embeddings: Embeddings, client: ClientAPI, collection: str) -> None:
        self.collection = collection
        self._client = client
        self._store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorMemory":
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model, base_url=settings.ollama_base_url
        )
        return cls(embeddings, client, settings.memory_collection)

    async def add(
        self, texts: Sequence[str], metadata: Mapping[str, MetadataValue] | None = None
    ) -> list[str]:
        created_at = datetime.now(UTC).isoformat()
        metadatas = [{"created_at": created_at, **(metadata or {})} for _ in texts]
        ids = [uuid4().hex for _ in texts]
        return await self._store.aadd_texts(list(texts), metadatas=metadatas, ids=ids)

    async def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[MemoryHit]:
        # Collection uses cosine distance, so similarity = 1 - distance (range [-1, 1]).
        results = await self._store.asimilarity_search_with_score(query, k=k)
        hits = [
            MemoryHit(id=doc.id, content=doc.page_content, score=1 - dist, metadata=doc.metadata)
            for doc, dist in results
        ]
        return [hit for hit in hits if hit.score >= min_score]

    async def delete(self, ids: Sequence[str]) -> None:
        await self._store.adelete(list(ids))

    def count(self) -> int:
        return self._client.get_collection(self.collection).count()
