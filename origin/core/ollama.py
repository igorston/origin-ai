"""Ollama server helpers: health probing and model warmup."""

import asyncio
import logging
import time

import httpx
from pydantic import BaseModel

from origin.config import Settings

logger = logging.getLogger(__name__)


class OllamaStatus(BaseModel):
    url: str
    reachable: bool
    models: dict[str, bool] = {}

    @property
    def healthy(self) -> bool:
        return self.reachable and all(self.models.values())


def _normalize(name: str) -> str:
    return name.removesuffix(":latest")


async def check_ollama(settings: Settings, timeout: float = 2.0) -> OllamaStatus:
    required = [settings.ollama_model, settings.ollama_embed_model]
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
    except httpx.HTTPError:
        return OllamaStatus(url=settings.ollama_base_url, reachable=False)
    installed = {_normalize(m["name"]) for m in response.json().get("models", [])}
    return OllamaStatus(
        url=settings.ollama_base_url,
        reachable=True,
        models={name: _normalize(name) in installed for name in required},
    )


async def warmup(settings: Settings) -> None:
    """Load the chat and embedding models into memory so the first request is fast."""
    requests = [
        ("/api/generate", {"model": settings.ollama_model}),
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
