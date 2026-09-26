from collections.abc import AsyncIterator
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from origin.core import ChatTurn, LLMEngine, ToolCallRecord

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = []
    use_memory: bool = True
    use_tools: bool = True


class ChatResponse(BaseModel):
    response: str
    model: str
    tool_calls: list[ToolCallRecord] = []


def get_engine(request: Request) -> LLMEngine:
    return request.app.state.engine


Engine = Annotated[LLMEngine, Depends(get_engine)]


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, engine: Engine) -> ChatResponse:
    result = await engine.generate(body.message, body.history, body.use_memory, body.use_tools)
    return ChatResponse(response=result.text, model=engine.model_name, tool_calls=result.tool_calls)


T = TypeVar("T")


async def prefetch(events: AsyncIterator[T]) -> AsyncIterator[T]:
    """Pull the first item before the response starts, so backend failures (e.g. Ollama
    down) surface as proper HTTP errors instead of a 200 with a broken stream."""
    first = await anext(events, None)

    async def replay() -> AsyncIterator[T]:
        if first is not None:
            yield first
        async for event in events:
            yield event

    return replay()


@router.post("/stream")
async def chat_stream(body: ChatRequest, engine: Engine) -> StreamingResponse:
    events = engine.stream(body.message, body.history, body.use_memory, body.use_tools)
    return StreamingResponse(await prefetch(events), media_type="text/plain; charset=utf-8")
