"""Accounts: users in the app's SQLite database, scrypt password hashes (stdlib only)."""

import asyncio
import base64
import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash  TEXT NOT NULL,
    is_admin       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);
"""
# scrypt cost: ~50 ms per check, 16 MiB of memory (OWASP's minimum is N=2^17 with r=8).
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, maxmem=2**26, **SCRYPT)
    b64 = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        check = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=32,
            maxmem=2**26,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(check, base64.b64decode(digest))


class User(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: str
    password_hash: str = ""

    @property
    def workspace(self) -> str:
        """The first user keeps the workspace Origin had before accounts existed."""
        return "" if self.id == 1 else f"u{self.id}"

    def public(self) -> dict:
        return {"id": self.id, "username": self.username, "is_admin": self.is_admin}


class UserError(ValueError):
    pass


class UserStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    # --- sync (also used by the CLI) ---------------------------------------------------

    def add(self, username: str, password: str, is_admin: bool = False) -> User:
        username = username.strip()
        if not username or len(username) > 64 or not username.isprintable():
            raise UserError("invalid username")
        if len(password) < MIN_PASSWORD:
            raise UserError(f"the password needs at least {MIN_PASSWORD} characters")
        try:
            with self._connect() as db:
                cursor = db.execute(
                    "INSERT INTO users (username, password_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        username,
                        hash_password(password),
                        int(is_admin),
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UserError(f"user {username!r} already exists") from exc
        return self.by_id(cursor.lastrowid)

    def by_id(self, user_id: int) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return User(**row) if row else None

    def by_name(self, username: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return User(**row) if row else None

    def all(self) -> list[User]:
        with self._connect() as db:
            return [User(**row) for row in db.execute("SELECT * FROM users ORDER BY id")]

    def count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def set_password(self, username: str, password: str) -> None:
        if len(password) < MIN_PASSWORD:
            raise UserError(f"the password needs at least {MIN_PASSWORD} characters")
        with self._connect() as db:
            changed = db.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (hash_password(password), username),
            ).rowcount
        if not changed:
            raise UserError(f"no user {username!r}")

    def remove(self, username: str) -> None:
        with self._connect() as db:
            if not db.execute("DELETE FROM users WHERE username = ?", (username,)).rowcount:
                raise UserError(f"no user {username!r}")

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.by_name(username.strip())
        if user is None:
            verify_password(password, _DUMMY_HASH)  # same timing whether or not it exists
            return None
        return user if verify_password(password, user.password_hash) else None

    # --- async wrappers for the server --------------------------------------------------

    async def aauthenticate(self, username: str, password: str) -> User | None:
        return await asyncio.to_thread(self.authenticate, username, password)

    async def aby_id(self, user_id: int) -> User | None:
        return await asyncio.to_thread(self.by_id, user_id)


_DUMMY_HASH = hash_password(secrets.token_hex(8))
