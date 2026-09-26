"""Origin — entry point.

Run:
    python main.py
    # or
    uvicorn main:app --reload
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

__version__ = "0.1.0"

logging.basicConfig(
    level=os.getenv("ORIGIN_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("origin")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Origin Core Initialized (v%s)", __version__)
    yield
    logger.info("Origin Core shutting down")


app = FastAPI(
    title="Origin",
    description="Local-first, modular personal AI assistant.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("ORIGIN_HOST", "127.0.0.1"),
        port=int(os.getenv("ORIGIN_PORT", "8000")),
        reload=os.getenv("ORIGIN_ENV", "dev") == "dev",
    )
