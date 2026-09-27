from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from origin.api.routes.chat import Context, Sessions
from origin.core.context import ContextManager, ContextUsage
from origin.memory.storage import Session, SessionStore, StoredMessage

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    title: str = ""


class SessionDetail(Session):
    messages: list[StoredMessage]
    context: ContextUsage


async def _require(sessions: SessionStore, session_id: str) -> Session:
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id!r} not found")
    return session


def session_context(
    session: Session, messages: list[StoredMessage], context: ContextManager
) -> ContextUsage:
    recent = [m for m in messages if m.id > session.summarized_upto]
    return context.usage(
        session.summary,
        recent,
        compactions=session.compactions,
        compressions=session.compressions,
        summarized=len(messages) - len(recent),
        closed=session.status == "closed",
        measured=session.context_tokens or None,
    )


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(sessions: Sessions, body: SessionCreate | None = None) -> Session:
    return await sessions.create((body or SessionCreate()).title)


@router.get("", response_model=list[Session])
async def list_sessions(
    sessions: Sessions, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> list[Session]:
    return await sessions.list(limit)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, sessions: Sessions, context: Context) -> SessionDetail:
    session = await _require(sessions, session_id)
    messages = await sessions.messages(session_id)
    return SessionDetail(
        **session.model_dump(),
        messages=messages,
        context=session_context(session, messages, context),
    )


@router.post("/{session_id}/continue", response_model=Session, status_code=status.HTTP_201_CREATED)
async def continue_session(session_id: str, sessions: Sessions, context: Context) -> Session:
    """Start a new conversation that carries this one's summary (typically after it was
    closed at the context limit), so nothing said so far is lost."""
    session = await _require(sessions, session_id)
    recent = await sessions.messages(session_id, after_id=session.summarized_upto)
    summary = await context.carry_over(session.summary, recent)
    title = session.title.removeprefix("↪ ")
    return await sessions.create(f"↪ {title}", parent_id=session.id, summary=summary)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, sessions: Sessions) -> None:
    if not await sessions.delete(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id!r} not found")
