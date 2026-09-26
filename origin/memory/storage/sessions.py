"""Conversation sessions persisted in a local SQLite database."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    tool_calls  TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
"""

TITLE_MAX_CHARS = 60

Role = Literal["user", "assistant"]


class Session(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class StoredMessage(BaseModel):
    role: Role
    content: str
    tool_calls: list[dict[str, Any]] = []
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    """Small synchronous SQLite store exposed through async methods (runs in a thread)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    # --- sync implementations -------------------------------------------------

    def _create(self, title: str) -> Session:
        now = _now()
        session = Session(
            id=uuid4().hex, title=title[:TITLE_MAX_CHARS], created_at=now, updated_at=now
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session.id, session.title, now, now),
            )
        return session

    def _get(self, session_id: str) -> Session | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
                " AS message_count FROM sessions s WHERE s.id = ?",
                (session_id,),
            ).fetchone()
        return Session(**row) if row else None

    def _list(self, limit: int) -> list[Session]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
                " AS message_count FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Session(**row) for row in rows]

    def _delete(self, session_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def _messages(self, session_id: str, limit: int | None) -> list[StoredMessage]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, tool_calls, created_at FROM messages"
                " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, -1 if limit is None else limit),
            ).fetchall()
        return [
            StoredMessage(
                role=row["role"],
                content=row["content"],
                tool_calls=json.loads(row["tool_calls"]),
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    def _append(
        self, session_id: str, role: Role, content: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(tool_calls, ensure_ascii=False), now),
            )
            # The first user message names an untitled session.
            db.execute(
                "UPDATE sessions SET updated_at = ?,"
                " title = CASE WHEN title = '' AND ? = 'user' THEN ? ELSE title END"
                " WHERE id = ?",
                (now, role, content[:TITLE_MAX_CHARS], session_id),
            )

    # --- async API ------------------------------------------------------------

    async def create(self, title: str = "") -> Session:
        return await asyncio.to_thread(self._create, title)

    async def get(self, session_id: str) -> Session | None:
        return await asyncio.to_thread(self._get, session_id)

    async def list(self, limit: int = 50) -> list[Session]:
        return await asyncio.to_thread(self._list, limit)

    async def delete(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._delete, session_id)

    async def messages(self, session_id: str, limit: int | None = None) -> list[StoredMessage]:
        """Most recent `limit` messages, in chronological order."""
        return await asyncio.to_thread(self._messages, session_id, limit)

    async def append(
        self,
        session_id: str,
        role: Role,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        await asyncio.to_thread(self._append, session_id, role, content, tool_calls or [])
