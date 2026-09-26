from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from origin.memory import MemoryHit, MetadataValue, VectorMemory

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    texts: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    metadata: dict[str, MetadataValue] = {}


class MemoryCreated(BaseModel):
    ids: list[str]


class MemoryStats(BaseModel):
    collection: str
    count: int


def get_memory(request: Request) -> VectorMemory:
    return request.app.state.memory


Memory = Annotated[VectorMemory, Depends(get_memory)]


@router.post("", response_model=MemoryCreated, status_code=status.HTTP_201_CREATED)
async def add_memory(body: MemoryCreate, memory: Memory) -> MemoryCreated:
    return MemoryCreated(ids=await memory.add(body.texts, body.metadata))


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


@router.get("/stats", response_model=MemoryStats)
async def memory_stats(memory: Memory) -> MemoryStats:
    return MemoryStats(collection=memory.collection, count=memory.count())
