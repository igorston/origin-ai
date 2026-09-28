"""Ollama server helpers: health probing and model warmup."""

import asyncio
import logging
import time
from collections.abc import Sequence

import httpx
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from origin.config import Settings
from origin.core.capacity import num_ctx
from origin.memory import VectorMemory

logger = logging.getLogger(__name__)


class OllamaStatus(BaseModel):
    url: str
    reachable: bool
    # Required model -> installed?
    models: dict[str, bool] = {}
    # Required model -> currently loaded in memory? (Ollama unloads idle models after
    # OLLAMA_KEEP_ALIVE; the next request then pays the load time, ~10-15s.)
    loaded: dict[str, bool] = {}

    @property
    def healthy(self) -> bool:
        return self.reachable and all(self.models.values())

    @property
    def ready(self) -> bool:
        return self.healthy and all(self.loaded.values())


def _normalize(name: str) -> str:
    return name.removesuffix(":latest")


async def check_ollama(settings: Settings, timeout: float = 2.0) -> OllamaStatus:
    required = [settings.ollama_model, settings.ollama_embed_model]
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout) as client:
            tags, running = await asyncio.gather(client.get("/api/tags"), client.get("/api/ps"))
            tags.raise_for_status()
    except httpx.HTTPError:
        return OllamaStatus(url=settings.ollama_base_url, reachable=False)
    installed = {_normalize(m["name"]) for m in tags.json().get("models", [])}
    in_memory = (
        {_normalize(m["name"]) for m in running.json().get("models", [])}
        if running.is_success
        else set()
    )
    return OllamaStatus(
        url=settings.ollama_base_url,
        reachable=True,
        models={name: _normalize(name) in installed for name in required},
        loaded={name: _normalize(name) in in_memory for name in required},
    )


async def warmup(
    settings: Settings,
    chat_models: Sequence[BaseChatModel] = (),
    memory: VectorMemory | None = None,
) -> None:
    """Load the models into Ollama, then make one tiny call through every client instance:
    each lazily builds its HTTP client on first use (~250ms on Windows), which otherwise
    lands on the first user requests."""
    await _load_models(settings)
    start = time.perf_counter()
    calls = [model.ainvoke("ok") for model in chat_models]
    if memory is not None:
        calls.append(memory.search("warmup", k=1))  # sync embedding client used by Chroma
    results = await asyncio.gather(*calls, return_exceptions=True)
    if failures := [r for r in results if isinstance(r, Exception)]:
        logger.warning("Client warmup failed: %s", failures[0])
    else:
        logger.info("Clients warmed up in %.1fs", time.perf_counter() - start)
    status = await check_ollama(settings)
    if status.healthy and not status.ready:
        # Ollama evicted one model to fit the other: every chat/embedding switch would
        # reload a model (measured +4-5s per call with num_ctx 8192 on an 8 GB GPU).
        logger.warning(
            "Chat and embedding models do not fit in memory together (loaded: %s). "
            'Lower OLLAMA_NUM_CTX (now %d), raise OLLAMA_VRAM_OVERHEAD_GB for "auto", '
            "or use smaller models.",
            ", ".join(name for name, up in status.loaded.items() if up) or "none",
            num_ctx(settings),
        )


async def _load_models(settings: Settings) -> None:
    requests = [
        # Same num_ctx as the chat clients, or the first real request reloads the model.
        (
            "/api/generate",
            {"model": settings.ollama_model, "options": {"num_ctx": num_ctx(settings)}},
        ),
        ("/api/embed", {"model": settings.ollama_embed_model, "input": "warmup"}),
    ]
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300) as client:
            responses = await asyncio.gather(
                *(
                    client.post(path, json={**body, "keep_alive": settings.ollama_keep_alive})
                    for path, body in requests
                )
            )
        for response in responses:
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Model warmup failed: %s", exc)
        return
    logger.info("Models warmed up in %.1fs", time.perf_counter() - start)
