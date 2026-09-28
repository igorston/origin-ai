import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from origin.branding import translator
from origin.core import ChatTurn, LLMEngine, ToolCallRecord, TurnUsage
from origin.core.context import (
    Compaction,
    ContextClosed,
    ContextManager,
    ContextUsage,
    MessageTooLong,
    Optimized,
    estimate_tokens,
)
from origin.memory.storage import Session, SessionStore, StoredMessage

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
    context: ContextUsage | None = None  # after this turn
    compactions: list[Compaction] = []  # context optimizations done before this turn
    trimmed: int = 0  # client-sent history: oldest messages dropped to fit


def get_engine(request: Request) -> LLMEngine:
    return request.app.state.engine


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


def get_context(request: Request) -> ContextManager:
    return request.app.state.context


Engine = Annotated[LLMEngine, Depends(get_engine)]
Sessions = Annotated[SessionStore, Depends(get_sessions)]
Context = Annotated[ContextManager, Depends(get_context)]


@dataclass
class PreparedTurn:
    history: list[ChatTurn]
    summary: str = ""
    session: Session | None = None
    compactions: list[Compaction] = field(default_factory=list)
    total_compactions: int = 0
    total_compressions: int = 0
    summarized: int = 0
    trimmed: int = 0
    before: ContextUsage | None = None
    scale: float = 1.0
    # Stored messages the model sees, with their token counts (for the context report).
    rows: list[StoredMessage] = field(default_factory=list)


@dataclass
class CountedTurn:
    role: str
    content: str
    tokens: int


def closed_error(reason: str, usage: ContextUsage | None) -> HTTPException:
    return HTTPException(
        status.HTTP_409_CONFLICT,
        detail={
            "message": translator()("chat.closed", reason=reason, id="{id}"),
            "closed": True,
            "reason": reason,
            "context": usage.model_dump() if usage else None,
        },
    )


async def persist_optimization(
    sessions: SessionStore, session_id: str, rows: list[StoredMessage], result: Optimized
) -> None:
    if not result.steps:
        return
    fields: dict = {
        "summary": result.summary,
        "compactions": result.compactions,
        "compressions": result.compressions,
        # The last measurement described the old layout; estimates take over until the
        # next turn measures again.
        "context_tokens": 0,
    }
    if folded := rows[: len(rows) - len(result.turns)]:
        fields["summarized_upto"] = folded[-1].id
    await sessions.update(session_id, **fields)


async def prepare_turn(
    body: ChatRequest, sessions: SessionStore, context: ContextManager
) -> PreparedTurn:
    """Fit the conversation into the context window before the model sees it."""
    if body.session_id is None:
        turns, trimmed = context.trim(list(body.history), body.message)
        return PreparedTurn(
            history=turns, trimmed=trimmed, before=context.usage("", turns, body.message)
        )

    session = await sessions.get(body.session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {body.session_id!r} not found")
    if session.status == "closed":
        raise closed_error(session.closed_reason, None)

    rows = await sessions.messages(session.id, after_id=session.summarized_upto)
    summarized = session.message_count - len(rows)
    try:
        result = await context.optimize(
            session.summary,
            rows,
            body.message,
            session.compactions,
            session.compressions,
            summarized,
            measured=session.context_tokens or None,
        )
    except MessageTooLong as exc:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            translator()("chat.too_long", tokens=exc.tokens, limit=exc.limit),
        ) from exc
    except ContextClosed as exc:
        # Keep the optimization done before giving up: a continuation starts from it.
        if exc.partial is not None:
            await persist_optimization(sessions, session.id, rows, exc.partial)
        await sessions.update(session.id, status="closed", closed_reason=exc.reason)
        raise closed_error(exc.reason, exc.usage) from exc

    kept = result.turns
    await persist_optimization(sessions, session.id, rows, result)
    return PreparedTurn(
        history=[ChatTurn(role=m.role, content=m.content) for m in kept],
        summary=result.summary,
        session=session,
        compactions=result.steps,
        total_compactions=result.compactions,
        total_compressions=result.compressions,
        summarized=result.summarized,
        before=context.usage(
            result.summary,
            kept,
            body.message,
            compactions=result.compactions,
            compressions=result.compressions,
            summarized=result.summarized,
            scale=result.scale,
        ),
        scale=result.scale,
        rows=list(kept),
    )


async def finish_turn(
    turn: PreparedTurn,
    body: ChatRequest,
    sessions: SessionStore,
    context: ContextManager,
    answer: str,
    tool_calls: list[ToolCallRecord],
    usage: TurnUsage | None,
) -> ContextUsage:
    """Persist the exchange (if any) and report the context after the turn."""
    user_tokens = estimate_tokens(body.message)
    answer_tokens = (
        usage.output_tokens if usage and usage.output_tokens else estimate_tokens(answer)
    )
    # input_tokens=0: the answer came from prompts other than the conversation (the writer),
    # so there is no measurement of the context to keep.
    measured = usage.input_tokens + usage.output_tokens if usage and usage.input_tokens else None
    if turn.session is not None:
        await sessions.append(turn.session.id, "user", body.message, tokens=user_tokens)
        await sessions.append(
            turn.session.id,
            "assistant",
            answer,
            [call.model_dump() for call in tool_calls],
            tokens=answer_tokens,
        )
        # Without a fresh measurement the old one no longer describes the context.
        await sessions.update(turn.session.id, context_tokens=measured or 0)
    # Real token counts where known: estimating a 2000-word story from its characters
    # reported the context at 112% when it was far below that.
    history = [
        *(turn.rows or turn.history),
        CountedTurn("user", body.message, user_tokens),
        CountedTurn("assistant", answer, answer_tokens),
    ]
    return context.usage(
        turn.summary,
        history,
        compactions=turn.total_compactions,
        compressions=turn.total_compressions,
        summarized=turn.summarized,
        measured=measured,
        scale=turn.scale,
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
    body: ChatRequest, engine: Engine, sessions: Sessions, context: Context
) -> ChatResponse:
    turn = await prepare_turn(body, sessions, context)
    result = await engine.generate(
        body.message, turn.history, body.use_memory, body.use_tools, turn.summary
    )
    after = await finish_turn(
        turn, body, sessions, context, result.text, result.tool_calls, result.usage
    )
    return ChatResponse(
        response=result.text,
        model=engine.model_name,
        session_id=body.session_id,
        tool_calls=result.tool_calls,
        context=after,
        compactions=turn.compactions,
        trimmed=turn.trimmed,
    )


class TurnDone(BaseModel):
    context: ContextUsage


async def run_and_persist(
    turn: PreparedTurn,
    body: ChatRequest,
    engine: LLMEngine,
    sessions: SessionStore,
    context: ContextManager,
) -> AsyncIterator[str | ToolCallRecord | TurnDone]:
    """Relay engine events, then save the exchange and report the context."""
    parts: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    usage: TurnUsage | None = None
    finished = False
    try:
        async for event in engine.events(
            body.message, turn.history, body.use_memory, body.use_tools, turn.summary
        ):
            if isinstance(event, TurnUsage):
                usage = event
                continue
            if isinstance(event, str):
                parts.append(event)
            else:
                tool_calls.append(event)
            yield event
        after = await finish_turn(turn, body, sessions, context, "".join(parts), tool_calls, usage)
        finished = True
        yield TurnDone(context=after)
    finally:
        # Also runs when the client disconnects mid-stream: keep what was produced.
        if not finished and (parts or tool_calls):
            await finish_turn(turn, body, sessions, context, "".join(parts), tool_calls, usage)


@router.post("/stream")
async def chat_stream(
    body: ChatRequest, engine: Engine, sessions: Sessions, context: Context
) -> StreamingResponse:
    """Plain-text token stream (easy to consume with curl)."""
    turn = await prepare_turn(body, sessions, context)

    async def text_only() -> AsyncIterator[str]:
        async for event in run_and_persist(turn, body, engine, sessions, context):
            if isinstance(event, str):
                yield event

    return StreamingResponse(await prefetch(text_only()), media_type="text/plain; charset=utf-8")


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/events")
async def chat_events(
    body: ChatRequest, engine: Engine, sessions: Sessions, context: Context
) -> StreamingResponse:
    """Server-Sent Events: `context` (budget, and any optimization done), `tool_call`,
    `token`, then `done` with the context after the turn (or `error` mid-stream)."""
    turn = await prepare_turn(body, sessions, context)
    events = await prefetch(run_and_persist(turn, body, engine, sessions, context))

    async def encode() -> AsyncIterator[str]:
        yield sse(
            "context",
            {
                "usage": turn.before.model_dump() if turn.before else None,
                "compactions": [c.model_dump() for c in turn.compactions],
                "trimmed": turn.trimmed,
            },
        )
        parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        after = None
        try:
            async for event in events:
                if isinstance(event, str):
                    parts.append(event)
                    yield sse("token", {"text": event})
                elif isinstance(event, TurnDone):
                    after = event.context
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
            context=after,
            compactions=turn.compactions,
            trimmed=turn.trimmed,
        )
        yield sse("done", done.model_dump())

    return StreamingResponse(
        encode(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
