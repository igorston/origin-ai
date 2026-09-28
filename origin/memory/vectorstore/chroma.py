import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

from origin.config import Settings
from origin.retry import RetryPolicy, ollama_client_kwargs

MetadataValue = str | int | float | bool


class MemoryHit(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, MetadataValue]


class MemoryRecord(BaseModel):
    id: str
    content: str
    metadata: dict[str, MetadataValue]


class VectorMemory:
    """Long-term semantic memory backed by a local Chroma collection (cosine similarity)."""

    def __init__(
        self,
        embeddings: Embeddings,
        client: ClientAPI,
        collection: str,
        dedup_threshold: float = 0.92,
    ) -> None:
        self.collection = collection
        # Paraphrases of a stored fact score >= ~0.90 with bge-m3, contradictions ~0.74-0.87.
        # Recalibrate (origin.memory.calibration) when the embedding model changes.
        self.dedup_threshold = dedup_threshold
        self.embeddings = embeddings
        self._client = client
        self._store = self._open()
        self._current_dim: int | None = None

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            client=self._client,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------ model compatibility

    def stored_dim(self) -> int | None:
        """Dimension of the vectors already stored (None if the collection is empty)."""
        stored = self._client.get_collection(self.collection).get(limit=1, include=["embeddings"])
        vectors = stored["embeddings"]
        return len(vectors[0]) if vectors is not None and len(vectors) else None

    async def current_dim(self) -> int:
        """Dimension produced by the current embedding model."""
        if self._current_dim is None:
            self._current_dim = len(await self.embeddings.aembed_query("dimension probe"))
        return self._current_dim

    def export(self) -> list[MemoryRecord]:
        """Every memory, archived included, for backups."""
        return self.records(limit=1_000_000)

    async def reindex(self, batch: int = 64) -> int:
        """Re-embed every memory with the current embedding model (required after changing
        it: old vectors have another dimension or live in another space). All new vectors
        are computed before the old collection is dropped, so a failure midway (e.g.
        Ollama down) leaves the memories untouched."""
        records = self.export()
        texts = [r.content for r in records]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors += await self.embeddings.aembed_documents(texts[start : start + batch])

        self._client.delete_collection(self.collection)
        self._store = self._open()
        self._current_dim = None
        collection = self._client.get_collection(self.collection)
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            collection.add(
                ids=[r.id for r in chunk],
                documents=[r.content for r in chunk],
                embeddings=vectors[start : start + batch],
                metadatas=[r.metadata or None for r in chunk],
            )
        return len(records)

    def sibling(self, collection: str) -> "VectorMemory":
        """Another collection on the same client and embeddings (one per workspace)."""
        return type(self)(self.embeddings, self._client, collection, self.dedup_threshold)

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorMemory":
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            keep_alive=settings.ollama_keep_alive,
            **ollama_client_kwargs(RetryPolicy.from_settings(settings)),
        )
        return cls(embeddings, client, settings.memory_collection, settings.memory_dedup_threshold)

    async def add(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, MetadataValue] | None = None,
        dedup: bool = True,
    ) -> list[str]:
        """Store texts and return their ids. With `dedup`, a text that near-duplicates an
        existing memory is not stored again; the existing memory's id is returned instead."""
        ids: list[str] = []
        for text in texts:
            if dedup and (existing := await self.find_duplicate(text)):
                ids.append(existing.id)
                continue
            metadatas = [{"created_at": datetime.now(UTC).isoformat(), **(metadata or {})}]
            ids.extend(await self._store.aadd_texts([text], metadatas=metadatas, ids=[uuid4().hex]))
        return ids

    async def find_duplicate(self, text: str) -> MemoryHit | None:
        hits = await self.search(text, k=1, min_score=self.dedup_threshold)
        return hits[0] if hits else None

    def records(self, limit: int = 100, offset: int = 0) -> list[MemoryRecord]:
        """All memories, newest first."""
        stored = self._store.get()
        records = [
            MemoryRecord(id=id_, content=text, metadata=meta or {})
            for id_, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"], strict=True
            )
        ]
        records.sort(key=lambda r: str(r.metadata.get("created_at", "")), reverse=True)
        return records[offset : offset + limit]

    def get(self, memory_id: str) -> MemoryRecord | None:
        stored = self._store.get(ids=[memory_id])
        if not stored["ids"]:
            return None
        return MemoryRecord(
            id=stored["ids"][0], content=stored["documents"][0], metadata=stored["metadatas"][0]
        )

    async def search(
        self, query: str, k: int = 4, min_score: float = 0.0, include_archived: bool = False
    ) -> list[MemoryHit]:
        # Collection uses cosine distance, so similarity = 1 - distance (range [-1, 1]).
        # `$ne` also matches memories stored before the `archived` flag existed.
        where = None if include_archived else {"archived": {"$ne": True}}
        results = await self._store.asimilarity_search_with_score(query, k=k, filter=where)
        hits = [
            MemoryHit(id=doc.id, content=doc.page_content, score=1 - dist, metadata=doc.metadata)
            for doc, dist in results
        ]
        return [hit for hit in hits if hit.score >= min_score]

    async def delete(self, ids: Sequence[str]) -> None:
        await self._store.adelete(list(ids))

    async def update(self, memory_id: str, content: str) -> MemoryRecord | None:
        """Replace a memory's text (re-embedding it); metadata is kept and marked edited."""
        record = self.get(memory_id)
        if record is None:
            return None
        metadata = {**record.metadata, "edited_at": datetime.now(UTC).isoformat()}
        document = Document(page_content=content, metadata=metadata, id=memory_id)
        await asyncio.to_thread(self._store.update_document, memory_id, document)
        return self.get(memory_id)

    def archive(self, memory_id: str, superseded_by: str = "") -> None:
        """Hide a memory from recall without destroying it (reversible with `restore`)."""
        self._set_flags(
            memory_id,
            archived=True,
            archived_at=datetime.now(UTC).isoformat(),
            superseded_by=superseded_by,
        )

    def restore(self, memory_id: str) -> bool:
        return self._set_flags(memory_id, archived=False, archived_at="", superseded_by="")

    def _set_flags(self, memory_id: str, **flags: MetadataValue) -> bool:
        record = self.get(memory_id)
        if record is None:
            return False
        self._client.get_collection(self.collection).update(
            ids=[memory_id], metadatas=[{**record.metadata, **flags}]
        )
        return True

    def count(self, include_archived: bool = True) -> int:
        collection = self._client.get_collection(self.collection)
        if include_archived:
            return collection.count()
        return len(collection.get(where={"archived": {"$ne": True}}, include=[])["ids"])
