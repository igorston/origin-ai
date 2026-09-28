"""Session cookies: `<payload>.<HMAC-SHA256>`, no server-side state.

The payload holds the user id, an expiry and a fingerprint of the password hash, so
changing a password signs every existing session out.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from pathlib import Path

from origin.auth.users import User
from origin.config import Settings

logger = logging.getLogger(__name__)

COOKIE = "origin_session"


def secret_key(settings: Settings) -> bytes:
    """ORIGIN_SECRET_KEY, else one generated once and kept next to the database."""
    if settings.origin_secret_key:
        return settings.origin_secret_key.encode()
    path = Path(settings.sqlite_path).with_name("secret.key")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
        logger.info("Generated a session signing key in %s", path)
    return path.read_text(encoding="utf-8").strip().encode()


def _fingerprint(user: User) -> str:
    return hashlib.sha256(user.password_hash.encode()).hexdigest()[:16]


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue(user: User, key: bytes, days: int) -> str:
    payload = {"uid": user.id, "exp": int(time.time()) + days * 86400, "pw": _fingerprint(user)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{signature}"


def read(token: str, key: bytes) -> dict | None:
    """The payload if the signature is valid and it has not expired."""
    try:
        body, signature = token.split(".")
        expected = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    return payload if payload.get("exp", 0) > time.time() else None


def still_valid(payload: dict, user: User | None) -> bool:
    return user is not None and payload.get("pw") == _fingerprint(user)
