"""Browser UI for manual testing: plain HTML/CSS/JS, no build step, no CDN (fully local)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def mount_web(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        # no-cache: always revalidate, so UI edits show up on a plain reload.
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
