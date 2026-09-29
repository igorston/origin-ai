"""The application: `create_app()` builds it, `run()` serves it.

Lives inside the package so that another project can depend on `origin-ai` and build the
same app, extended by its plugins (see `origin.plugins`).
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from origin import __version__
from origin.api.errors import register_error_handlers
from origin.api.routes import auth, calibration, chat, health, memory, sessions, setup, tools
from origin.auth import UserStore, bootstrap_admin
from origin.auth.middleware import AuthMiddleware
from origin.auth.tokens import secret_key
from origin.branding import get_brand
from origin.config import Settings, get_settings
from origin.core import LLMEngine, make_chat_model
from origin.core.capacity import capacity_for
from origin.core.context import ContextBudget, ContextManager, ConversationSummarizer
from origin.core.ollama import check_ollama, warmup
from origin.memory import VectorMemory
from origin.memory.storage import SessionStore
from origin.plugins import load_app_plugins
from origin.retry import RetryPolicy
from origin.web import mount_web
from origin.workspace import Workspaces

logger = logging.getLogger("origin")


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.origin_log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    brand = get_brand(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Window size from the model's limit and the free VRAM (OLLAMA_NUM_CTX=auto),
        # probed before any chat client is built: they all have to use the same value.
        # Skipped with a fixed window and no warmup (tests), which is when nothing may
        # touch Ollama.
        probing = settings.ollama_num_ctx == "auto" or settings.ollama_warmup
        capacity = await asyncio.to_thread(capacity_for, settings) if probing else None
        app.state.sessions = SessionStore(settings.sqlite_path)
        if settings.origin_auth != "off":
            app.state.users = UserStore(settings.sqlite_path)
            app.state.auth_key = secret_key(settings)
            bootstrap_admin(settings, app.state.users)
        judge = make_chat_model(settings, temperature=0)
        ollama = await check_ollama(settings)
        # One workspace per user when authentication is on; the default one otherwise.
        app.state.workspaces = Workspaces(
            settings, VectorMemory.from_settings(settings), judge, LLMEngine.from_settings(settings)
        )
        default = await app.state.workspaces.get(probe=ollama.healthy)
        engine = default.engine
        app.state.context = ContextManager(
            ContextBudget.from_settings(settings),
            ConversationSummarizer(
                judge, settings.context_summary_max_tokens, RetryPolicy.from_settings(settings)
            ),
            base_tokens=engine.estimate_base_tokens(),  # measured exactly after warmup
            capacity=capacity,
            model=settings.ollama_model,
        )
        logger.info(
            "%s initialized (v%s) | model=%s | embeddings=%s | memories=%d | tools=%s",
            brand.product_name,
            __version__,
            settings.ollama_model,
            settings.ollama_embed_model,
            default.memory.count(),
            ",".join(engine.tools) or "none",
        )

        if not ollama.reachable:
            logger.warning("Ollama is not reachable at %s; chat will fail until it is", ollama.url)
        elif missing := [name for name, ok in ollama.models.items() if not ok]:
            logger.warning(
                "Missing Ollama models: %s (run `ollama pull <model>`)", ", ".join(missing)
            )

        async def warm_and_measure() -> None:
            await warmup(settings, [engine.model, engine.router, judge], default.memory)
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
                logger.warning(
                    "Could not measure the fixed prompt; using an estimate", exc_info=True
                )

        warmup_task = None
        if settings.ollama_warmup and ollama.healthy:
            warmup_task = asyncio.create_task(warm_and_measure())
        yield
        if warmup_task:
            warmup_task.cancel()
        logger.info("%s shutting down", brand.product_name)

    app = FastAPI(
        title=brand.product_name,
        description=brand.description,
        version=__version__,
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.add_middleware(AuthMiddleware)
    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(calibration.router)
    app.include_router(memory.router)
    app.include_router(tools.router)
    app.include_router(setup.router)
    # Plugins see every built-in route, and may replace the web UI's by mounting their own
    # first (a route registered earlier wins).
    load_app_plugins(app, settings)
    mount_web(app)
    return app


def run(app: str = "origin.app:create_app", factory: bool = True) -> None:
    """Serve `app`, an import string: a factory returning the app (the default), or with
    `factory=False` the app object itself. An import string, so `ORIGIN_ENV=dev` can reload."""
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        app,
        factory=factory,
        host=settings.origin_host,
        port=settings.origin_port,
        reload=settings.origin_env == "dev",
    )
