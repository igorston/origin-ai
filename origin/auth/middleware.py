"""Who is calling: with ORIGIN_AUTH=password every request but the public ones needs a
valid session cookie, and gets its user's workspace. With "off" nothing changes."""

from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from origin.auth import tokens
from origin.auth.users import User
from origin.branding import translator
from origin.config import get_settings

# The login page and what it needs to render, plus health checks.
PUBLIC_PATHS = {"/login", "/api/auth/login", "/api/auth/me", "/api/brand", "/health"}
PUBLIC_PREFIXES = ("/static/", "/brand/")
ADMIN_ONLY = {("POST", "/setup/pull")}  # downloads gigabytes onto the server
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


async def current_user(request: Request) -> User | None:
    token = request.cookies.get(tokens.COOKIE)
    payload = tokens.read(token, request.app.state.auth_key) if token else None
    if payload is None:
        return None
    user = await request.app.state.users.aby_id(payload["uid"])
    return user if tokens.still_valid(payload, user) else None


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.user = None
        if get_settings().origin_auth == "off":
            return await call_next(request)
        t = translator()
        # Cookies ride along on cross-site requests; browsers always send Origin on them.
        origin = request.headers.get("origin")
        if request.method in UNSAFE_METHODS and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": t("auth.cross_origin")}, status_code=403)

        user = await current_user(request)
        path = request.url.path
        if user is None and path not in PUBLIC_PATHS and not path.startswith(PUBLIC_PREFIXES):
            if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                return RedirectResponse("/login", status_code=303)
            return JSONResponse({"detail": t("auth.required")}, status_code=401)
        if user and (request.method, path) in ADMIN_ONLY and not user.is_admin:
            return JSONResponse({"detail": t("auth.admin_only")}, status_code=403)

        request.state.user = user
        if user is not None:
            request.state.workspace = await request.app.state.workspaces.get(user.workspace)
        return await call_next(request)
