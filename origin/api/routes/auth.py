"""Sign in / out (ORIGIN_AUTH=password).

Two ways in, one logic: POST /api/auth/login (JSON, used by the login page's script)
and POST /login (a plain form post, so signing in works before the script loads, or
without it, and the password never ends up in a URL).
"""

import time
from collections import defaultdict, deque
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from origin.auth import tokens
from origin.auth.middleware import current_user
from origin.auth.users import User
from origin.branding import translator
from origin.config import get_settings

router = APIRouter(tags=["auth"])

# Failed logins per client address: at most MAX_FAILURES in WINDOW seconds.
MAX_FAILURES = 10
WINDOW = 600
_failures: dict[str, deque[float]] = defaultdict(deque)


class Credentials(BaseModel):
    username: str
    password: str


def _client(request: Request) -> str:
    return request.client.host if request.client else "?"


def _too_many(address: str) -> bool:
    attempts = _failures[address]
    while attempts and attempts[0] < time.monotonic() - WINDOW:
        attempts.popleft()
    return len(attempts) >= MAX_FAILURES


async def authenticate(request: Request, username: str, password: str) -> User:
    """The user, or HTTPException 401 / 429 (brute force is throttled per address)."""
    t = translator()
    if get_settings().origin_auth == "off":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "authentication is off")
    address = _client(request)
    if _too_many(address):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, t("auth.too_many"))
    user = await request.app.state.users.aauthenticate(username, password)
    if user is None:
        _failures[address].append(time.monotonic())
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, t("auth.failed"))
    _failures.pop(address, None)
    return user


def set_session(response: Response, request: Request, user: User) -> None:
    settings = get_settings()
    response.set_cookie(
        tokens.COOKIE,
        tokens.issue(user, request.app.state.auth_key, settings.origin_session_days),
        max_age=settings.origin_session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.origin_cookie_secure or request.url.scheme == "https",
        path="/",
    )


@router.post("/api/auth/login")
async def login(body: Credentials, request: Request, response: Response) -> dict:
    user = await authenticate(request, body.username, body.password)
    set_session(response, request, user)
    return user.public()


@router.post("/login", include_in_schema=False)
async def login_form(request: Request) -> RedirectResponse:
    form = parse_qs((await request.body()).decode("utf-8", "replace"))
    field = lambda name: (form.get(name) or [""])[0]  # noqa: E731
    try:
        user = await authenticate(request, field("username"), field("password"))
    except HTTPException as exc:
        reason = "throttled" if exc.status_code == 429 else "failed"
        return RedirectResponse(f"/login?error={reason}", status_code=303)
    response = RedirectResponse("/", status_code=303)
    set_session(response, request, user)
    return response


@router.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    response.delete_cookie(tokens.COOKIE, path="/")


@router.get("/api/auth/me")
async def me(request: Request) -> dict:
    """The authentication mode and the signed-in user (null when signed out or off)."""
    mode = get_settings().origin_auth
    user = await current_user(request) if mode != "off" else None
    return {"auth": mode, "user": user.public() if user else None}
