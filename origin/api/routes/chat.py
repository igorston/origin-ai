from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from origin.core import ChatTurn, LLMEngine

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    response: str
    model: str


def get_engine(request: Request) -> LLMEngine:
    return request.app.state.engine


Engine = Annotated[LLMEngine, Depends(get_engine)]


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, engine: Engine) -> ChatResponse:
    response = await engine.generate(body.message, body.history)
    return ChatResponse(response=response, model=engine.model_name)


@router.post("/stream")
async def chat_stream(body: ChatRequest, engine: Engine) -> StreamingResponse:
    return StreamingResponse(
        engine.stream(body.message, body.history), media_type="text/plain; charset=utf-8"
    )
