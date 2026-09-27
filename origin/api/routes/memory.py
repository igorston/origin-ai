from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, model_validator

from origin.memory import MemoryHit, MemoryRecord, MetadataValue, VectorMemory

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    texts: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    metadata: dict[str, MetadataValue] = {}


class MemoryCreated(BaseModel):
    # Near-duplicates of stored memories are not stored again; their existing id is returned.
    ids: list[str]


class MemoryUpdate(BaseModel):
    content: Annotated[str, Field(min_length=1)] | None = None
    archived: bool | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> "MemoryUpdate":
        if self.content is None and self.archived is None:
            raise ValueError("send content and/or archived")
        return self


class MemoryStats(BaseModel):
    collection: str
    count: int
    active: int
    archived: int


def get_memory(request: Request) -> VectorMemory:
    return request.app.state.memory


Memory = Annotated[VectorMemory, Depends(get_memory)]


@router.post("", response_model=MemoryCreated, status_code=status.HTTP_201_CREATED)
async def add_memory(body: MemoryCreate, memory: Memory) -> MemoryCreated:
    return MemoryCreated(ids=await memory.add(body.texts, body.metadata))


@router.get("", response_model=list[MemoryRecord])
async def list_memories(
    memory: Memory,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[MemoryRecord]:
    """All memories, newest first."""
    return memory.records(limit=limit, offset=offset)


@router.get("/search", response_model=list[MemoryHit])
async def search_memory(
    memory: Memory,
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=50)] = 4,
    min_score: Annotated[float, Query(ge=0, le=1)] = 0.0,
) -> list[MemoryHit]:
    return await memory.search(q, k=k, min_score=min_score)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: str, memory: Memory) -> None:
    await memory.delete([memory_id])


@router.patch("/{memory_id}", response_model=MemoryRecord)
async def update_memory(memory_id: str, body: MemoryUpdate, memory: Memory) -> MemoryRecord:
    """Edit a memory's text (it is re-embedded) and/or archive/unarchive it."""
    if memory.get(memory_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Memory {memory_id!r} not found")
    if body.content is not None:
        await memory.update(memory_id, body.content.strip())
    if body.archived is True:
        memory.archive(memory_id)
    elif body.archived is False:
        memory.restore(memory_id)
    return memory.get(memory_id)


@router.post("/{memory_id}/restore", response_model=MemoryRecord)
async def restore_memory(memory_id: str, memory: Memory) -> MemoryRecord:
    """Bring back a memory that was archived because a newer fact superseded it."""
    if not memory.restore(memory_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Memory {memory_id!r} not found")
    return memory.get(memory_id)


@router.get("/stats", response_model=MemoryStats)
async def memory_stats(memory: Memory) -> MemoryStats:
    total = memory.count()
    active = memory.count(include_archived=False)
    return MemoryStats(
        collection=memory.collection, count=total, active=active, archived=total - active
    )
