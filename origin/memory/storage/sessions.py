"""Conversation sessions persisted in a local SQLite database."""

import asyncio
import builtins  # the `list` method below shadows the builtin in later annotations (<3.14)
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

# Columns added after the first release; applied to existing databases on open.
MIGRATIONS = {
    "sessions": [
        # Running summary of the messages up to `summarized_upto` (a message id), which
        # the model sees instead of those messages once the context gets full.
        ("summary", "TEXT NOT NULL DEFAULT ''"),
        ("summarized_upto", "INTEGER NOT NULL DEFAULT 0"),
        ("compactions", "INTEGER NOT NULL DEFAULT 0"),
        ("compressions", "INTEGER NOT NULL DEFAULT 0"),
        ("context_tokens", "INTEGER NOT NULL DEFAULT 0"),  # measured after the last turn
        ("status", "TEXT NOT NULL DEFAULT 'open'"),
        ("closed_reason", "TEXT NOT NULL DEFAULT ''"),
        ("parent_id", "TEXT NOT NULL DEFAULT ''"),  # session this one continues
    ],
    "messages": [("tokens", "INTEGER NOT NULL DEFAULT 0")],
}

TITLE_MAX_CHARS = 60

Role = Literal["user", "assistant"]
Status = Literal["open", "closed"]


class Session(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    summary: str = ""
    summarized_upto: int = 0
    compactions: int = 0
    compressions: int = 0
    context_tokens: int = 0
    status: Status = "open"
    closed_reason: str = ""
    parent_id: str = ""


class StoredMessage(BaseModel):
    id: int = 0
    role: Role
    content: str
    tool_calls: list[dict[str, Any]] = []
    tokens: int = 0
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


UPDATABLE = {
    "summary",
    "summarized_upto",
    "compactions",
    "compressions",
    "context_tokens",
    "status",
    "closed_reason",
}


class SessionStore:
    """Small synchronous SQLite store exposed through async methods (runs in a thread)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            for table, columns in MIGRATIONS.items():
                existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                for name, definition in columns:
                    if name not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

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

    def _create(self, title: str, parent_id: str, summary: str) -> Session:
        now = _now()
        session = Session(
            id=uuid4().hex,
            title=title[:TITLE_MAX_CHARS],
            created_at=now,
            updated_at=now,
            parent_id=parent_id,
            summary=summary,
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, parent_id, summary)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session.id, session.title, now, now, parent_id, summary),
            )
        return session

    _SELECT = (
        "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
        " AS message_count FROM sessions s"
    )

    def _get(self, session_id: str) -> Session | None:
        with self._connect() as db:
            row = db.execute(f"{self._SELECT} WHERE s.id = ?", (session_id,)).fetchone()
        return Session(**row) if row else None

    def _list(self, limit: int) -> list[Session]:
        with self._connect() as db:
            rows = db.execute(
                f"{self._SELECT} ORDER BY s.updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Session(**row) for row in rows]

    def _delete(self, session_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def _messages(self, session_id: str, limit: int | None, after_id: int) -> list[StoredMessage]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, role, content, tool_calls, tokens, created_at FROM messages"
                " WHERE session_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
                (session_id, after_id, -1 if limit is None else limit),
            ).fetchall()
        return [
            StoredMessage(**{**row, "tool_calls": json.loads(row["tool_calls"])})
            for row in reversed(rows)
        ]

    def _append(
        self,
        session_id: str,
        role: Role,
        content: str,
        tool_calls: list[dict[str, Any]],
        tokens: int,
    ) -> int:
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False),
                    tokens,
                    now,
                ),
            )
            # The first user message names an untitled session.
            db.execute(
                "UPDATE sessions SET updated_at = ?,"
                " title = CASE WHEN title = '' AND ? = 'user' THEN ? ELSE title END"
                " WHERE id = ?",
                (now, role, content[:TITLE_MAX_CHARS], session_id),
            )
            return cursor.lastrowid

    def _update(self, session_id: str, fields: dict[str, Any]) -> None:
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"Not updatable: {sorted(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as db:
            db.execute(
                f"UPDATE sessions SET {assignments} WHERE id = ?", (*fields.values(), session_id)
            )

    # --- async API ------------------------------------------------------------

    async def create(self, title: str = "", parent_id: str = "", summary: str = "") -> Session:
        return await asyncio.to_thread(self._create, title, parent_id, summary)

    async def get(self, session_id: str) -> Session | None:
        return await asyncio.to_thread(self._get, session_id)

    async def list(self, limit: int = 50) -> list[Session]:
        return await asyncio.to_thread(self._list, limit)

    async def delete(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._delete, session_id)

    async def messages(
        self, session_id: str, limit: int | None = None, after_id: int = 0
    ) -> builtins.list[StoredMessage]:
        """Most recent `limit` messages with id > `after_id`, in chronological order."""
        return await asyncio.to_thread(self._messages, session_id, limit, after_id)

    async def append(
        self,
        session_id: str,
        role: Role,
        content: str,
        tool_calls: builtins.list[dict[str, Any]] | None = None,
        tokens: int = 0,
    ) -> int:
        return await asyncio.to_thread(
            self._append, session_id, role, content, tool_calls or [], tokens
        )

    async def update(self, session_id: str, **fields: Any) -> None:
        await asyncio.to_thread(self._update, session_id, fields)
