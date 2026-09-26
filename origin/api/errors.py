"""Map model-backend failures to clear HTTP errors instead of generic 500s."""

import logging

import httpx
import ollama
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from origin.config import get_settings

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    async def ollama_unreachable(request: Request, exc: Exception) -> JSONResponse:
        url = get_settings().ollama_base_url
        logger.error("Ollama unreachable at %s: %s", url, exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"Ollama is not reachable at {url}. Is `ollama serve` running?"},
        )

    async def ollama_error(request: Request, exc: ollama.ResponseError) -> JSONResponse:
        logger.error("Ollama error (%s): %s", exc.status_code, exc.error)
        detail = f"Ollama error: {exc.error}"
        if exc.status_code == 404:
            detail += " — install it with `ollama pull <model>`."
        return JSONResponse(status_code=502, content={"detail": detail})

    app.add_exception_handler(httpx.ConnectError, ollama_unreachable)
    app.add_exception_handler(ConnectionError, ollama_unreachable)
    app.add_exception_handler(ollama.ResponseError, ollama_error)
