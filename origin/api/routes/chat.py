import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from origin.config import Settings, get_settings
from origin.core import ChatTurn, LLMEngine, ToolCallRecord
from origin.memory.storage import SessionStore

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    # Either keep the conversation server-side with `session_id`, or send `history` yourself.
    session_id: str | None = None
    history: list[ChatTurn] = []
    use_memory: bool = True
    use_tools: bool = True

    @model_validator(mode="after")
    def _session_or_history(self) -> "ChatRequest":
        if self.session_id and self.history:
            raise ValueError("send either session_id or history, not both")
        return self


class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: str | None = None
    tool_calls: list[ToolCallRecord] = []


def get_engine(request: Request) -> LLMEngine:
    return request.app.state.engine


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


Engine = Annotated[LLMEngine, Depends(get_engine)]
Sessions = Annotated[SessionStore, Depends(get_sessions)]
AppSettings = Annotated[Settings, Depends(get_settings)]


async def resolve_history(body: ChatRequest, sessions: SessionStore, limit: int) -> list[ChatTurn]:
    if body.session_id is None:
        return body.history
    if await sessions.get(body.session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {body.session_id!r} not found")
    stored = await sessions.messages(body.session_id, limit=limit)
    return [ChatTurn(role=m.role, content=m.content) for m in stored]


async def save_exchange(
    sessions: SessionStore,
    session_id: str | None,
    message: str,
    answer: str,
    tool_calls: list[ToolCallRecord],
) -> None:
    if session_id is None:
        return
    await sessions.append(session_id, "user", message)
    await sessions.append(
        session_id, "assistant", answer, [call.model_dump() for call in tool_calls]
    )


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


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest, engine: Engine, sessions: Sessions, settings: AppSettings
) -> ChatResponse:
    history = await resolve_history(body, sessions, settings.session_history_limit)
    result = await engine.generate(body.message, history, body.use_memory, body.use_tools)
    await save_exchange(sessions, body.session_id, body.message, result.text, result.tool_calls)
    return ChatResponse(
        response=result.text,
        model=engine.model_name,
        session_id=body.session_id,
        tool_calls=result.tool_calls,
    )


async def run_and_persist(
    body: ChatRequest, engine: LLMEngine, sessions: SessionStore, history: list[ChatTurn]
) -> AsyncIterator[str | ToolCallRecord]:
    """Relay engine events and save the exchange to the session once the turn ends."""
    parts: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    try:
        async for event in engine.events(body.message, history, body.use_memory, body.use_tools):
            if isinstance(event, str):
                parts.append(event)
            else:
                tool_calls.append(event)
            yield event
    finally:
        # Also runs when the client disconnects mid-stream: keep what was produced.
        if parts or tool_calls:
            await save_exchange(sessions, body.session_id, body.message, "".join(parts), tool_calls)


@router.post("/stream")
async def chat_stream(
    body: ChatRequest, engine: Engine, sessions: Sessions, settings: AppSettings
) -> StreamingResponse:
    """Plain-text token stream (easy to consume with curl)."""
    history = await resolve_history(body, sessions, settings.session_history_limit)

    async def text_only() -> AsyncIterator[str]:
        async for event in run_and_persist(body, engine, sessions, history):
            if isinstance(event, str):
                yield event

    return StreamingResponse(await prefetch(text_only()), media_type="text/plain; charset=utf-8")


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/events")
async def chat_events(
    body: ChatRequest, engine: Engine, sessions: Sessions, settings: AppSettings
) -> StreamingResponse:
    """Server-Sent Events: `tool_call`, `token`, then `done` (or `error` mid-stream)."""
    history = await resolve_history(body, sessions, settings.session_history_limit)
    events = await prefetch(run_and_persist(body, engine, sessions, history))

    async def encode() -> AsyncIterator[str]:
        parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        try:
            async for event in events:
                if isinstance(event, str):
                    parts.append(event)
                    yield sse("token", {"text": event})
                else:
                    tool_calls.append(event)
                    yield sse("tool_call", event.model_dump())
        except Exception as exc:  # headers are already sent: report in-band
            logger.exception("Chat stream failed")
            yield sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
            return
        done = ChatResponse(
            response="".join(parts),
            model=engine.model_name,
            session_id=body.session_id,
            tool_calls=tool_calls,
        )
        yield sse("done", done.model_dump())

    return StreamingResponse(
        encode(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
