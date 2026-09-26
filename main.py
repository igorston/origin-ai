"""Origin — entry point.

Run:
    python main.py
    # or
    uvicorn main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from origin import __version__
from origin.api.routes import chat, health, memory, tools
from origin.config import get_settings
from origin.core import LLMEngine
from origin.integrations import ToolContext, ToolRegistry
from origin.memory import VectorMemory

settings = get_settings()

logging.basicConfig(
    level=settings.origin_log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("origin")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.memory = VectorMemory.from_settings(settings)
    registry = (
        ToolRegistry.discover(
            ToolContext(settings, app.state.memory), disabled=settings.tools_disabled
        )
        if settings.tools_enabled
        else ToolRegistry()
    )
    app.state.engine = LLMEngine.from_settings(
        settings, memory=app.state.memory, tools=registry.tools
    )
    logger.info(
        "Origin Core Initialized (v%s) | model=%s | embeddings=%s | memories=%d | tools=%s",
        __version__,
        settings.ollama_model,
        settings.ollama_embed_model,
        app.state.memory.count(),
        ",".join(registry.tools) or "none",
    )
    yield
    logger.info("Origin Core shutting down")


app = FastAPI(
    title="Origin",
    description="Local-first, modular personal AI assistant.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(tools.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.origin_host,
        port=settings.origin_port,
        reload=settings.origin_env == "dev",
    )
