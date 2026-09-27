"""Origin — entry point.

Run:
    python main.py
    # or
    uvicorn main:app --reload
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from origin import __version__
from origin.api.errors import register_error_handlers
from origin.api.routes import calibration, chat, health, memory, sessions, tools
from origin.config import get_settings
from origin.core import LLMEngine, make_chat_model
from origin.core.context import ContextBudget, ContextManager, ConversationSummarizer
from origin.core.ollama import check_ollama, warmup
from origin.integrations import ToolContext, ToolRegistry
from origin.memory import VectorMemory
from origin.memory.calibration import CalibrationStore, MemoryCalibrator, Thresholds
from origin.memory.curator import MemoryCurator
from origin.memory.storage import SessionStore
from origin.retry import RetryPolicy
from origin.web import mount_web

settings = get_settings()

logging.basicConfig(
    level=settings.origin_log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("origin")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.memory = VectorMemory.from_settings(settings)
    app.state.sessions = SessionStore(settings.sqlite_path)
    judge = make_chat_model(settings, temperature=0)
    app.state.curator = MemoryCurator(
        app.state.memory,
        judge,
        settings.memory_conflict_threshold,
        retry=RetryPolicy.from_settings(settings),
    )
    registry = (
        ToolRegistry.discover(
            ToolContext(settings, app.state.memory, llm=judge, curator=app.state.curator),
            disabled=settings.tools_disabled,
        )
        if settings.tools_enabled
        else ToolRegistry()
    )
    engine = LLMEngine.from_settings(settings, memory=app.state.memory, tools=registry.tools)
    app.state.engine = engine

    def apply_thresholds(thresholds: Thresholds) -> None:
        # Every component that reads a memory threshold, updated in place.
        app.state.memory.dedup_threshold = thresholds.dedup
        app.state.curator.conflict_threshold = thresholds.conflict
        engine.memory_min_score = thresholds.min_score

    app.state.calibrator = MemoryCalibrator(
        app.state.memory,
        judge=app.state.curator,
        store=CalibrationStore(settings.calibration_path),
        embed_model=settings.ollama_embed_model,
        chat_model=settings.ollama_model,
        defaults=Thresholds(
            dedup=settings.memory_dedup_threshold,
            conflict=settings.memory_conflict_threshold,
            min_score=settings.memory_min_score,
        ),
        apply=apply_thresholds,
        backup_dir=settings.memory_backup_dir,
    )
    app.state.context = ContextManager(
        ContextBudget.from_settings(settings),
        ConversationSummarizer(
            judge, settings.context_summary_max_tokens, RetryPolicy.from_settings(settings)
        ),
        base_tokens=engine.estimate_base_tokens(),  # measured exactly after warmup
    )
    logger.info(
        "Origin Core Initialized (v%s) | model=%s | embeddings=%s | memories=%d | tools=%s",
        __version__,
        settings.ollama_model,
        settings.ollama_embed_model,
        app.state.memory.count(),
        ",".join(registry.tools) or "none",
    )

    ollama = await check_ollama(settings)
    if not ollama.reachable:
        logger.warning("Ollama is not reachable at %s; chat will fail until it is", ollama.url)
    elif missing := [name for name, ok in ollama.models.items() if not ok]:
        logger.warning("Missing Ollama models: %s (run `ollama pull <model>`)", ", ".join(missing))
    await app.state.calibrator.bootstrap(probe=ollama.healthy)

    async def warm_and_measure() -> None:
        await warmup(settings, [engine.model, engine.router, judge], app.state.memory)
        try:
            app.state.context.base_tokens = await engine.measure_base_tokens()
            logger.info(
                "Context: window=%d tokens, fixed prompt=%d tokens",
                app.state.context.budget.window,
                app.state.context.base_tokens,
            )
            if problem := app.state.context.check_window():
                logger.warning(problem)
        except Exception:
            logger.warning("Could not measure the fixed prompt; using an estimate", exc_info=True)

    warmup_task = None
    if settings.ollama_warmup and ollama.healthy:
        warmup_task = asyncio.create_task(warm_and_measure())
    yield
    if warmup_task:
        warmup_task.cancel()
    logger.info("Origin Core shutting down")


app = FastAPI(
    title="Origin",
    description="Local-first, modular personal AI assistant.",
    version=__version__,
    lifespan=lifespan,
)
register_error_handlers(app)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(calibration.router)
app.include_router(memory.router)
app.include_router(tools.router)
mount_web(app)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.origin_host,
        port=settings.origin_port,
        reload=settings.origin_env == "dev",
    )
