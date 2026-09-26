from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from origin.api.routes.chat import Sessions
from origin.memory.storage import Session, StoredMessage

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    title: str = ""


class SessionDetail(Session):
    messages: list[StoredMessage]


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(sessions: Sessions, body: SessionCreate | None = None) -> Session:
    return await sessions.create((body or SessionCreate()).title)


@router.get("", response_model=list[Session])
async def list_sessions(
    sessions: Sessions, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> list[Session]:
    return await sessions.list(limit)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, sessions: Sessions) -> SessionDetail:
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id!r} not found")
    return SessionDetail(**session.model_dump(), messages=await sessions.messages(session_id))


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, sessions: Sessions) -> None:
    if not await sessions.delete(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id!r} not found")
