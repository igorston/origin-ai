"""First-run setup: download the models Ollama is missing, with progress (SSE)."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from origin.config import Settings, get_settings
from origin.core.ollama import OllamaStatus, check_ollama

router = APIRouter(prefix="/setup", tags=["setup"])
logger = logging.getLogger(__name__)

AppSettings = Annotated[Settings, Depends(get_settings)]


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/status", response_model=OllamaStatus)
async def status(settings: AppSettings) -> OllamaStatus:
    """Is Ollama reachable, and which required models are installed."""
    return await check_ollama(settings)


async def pull(settings: Settings, model: str) -> AsyncIterator[dict]:
    """Ollama's own progress lines for one model: {status, total?, completed?}."""
    timeout = httpx.Timeout(30, read=None)  # a pull streams for minutes
    async with (
        httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout) as client,
        client.stream("POST", "/api/pull", json={"model": model, "stream": True}) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.strip():
                yield json.loads(line)


@router.post("/pull")
async def pull_missing(settings: AppSettings) -> StreamingResponse:
    """Download every required model that is not installed yet. Events: `progress`
    {model, status, completed, total, percent}, `model_done` {model}, `done` {models},
    `error` {model, detail}."""
    state = await check_ollama(settings)
    missing = [name for name, installed in state.models.items() if not installed]

    async def events() -> AsyncIterator[str]:
        if not state.reachable:
            yield sse("error", {"model": None, "detail": f"Ollama unreachable at {state.url}"})
            return
        for model in missing:
            logger.info("Setup: pulling %s", model)
            try:
                async for line in pull(settings, model):
                    if "error" in line:
                        raise RuntimeError(line["error"])
                    total, completed = line.get("total"), line.get("completed")
                    percent = round(completed / total * 100) if total and completed else None
                    yield sse(
                        "progress",
                        {
                            "model": model,
                            "status": line.get("status", ""),
                            "completed": completed,
                            "total": total,
                            "percent": percent,
                        },
                    )
            except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
                logger.warning("Setup: pulling %s failed: %s", model, exc)
                yield sse("error", {"model": model, "detail": str(exc)})
                return
            yield sse("model_done", {"model": model})
        yield sse("done", {"models": missing})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
