"""The web interface in a real browser (Playwright), against the app served by uvicorn.

The model and Ollama are faked, so this runs anywhere: `pip install -e ".[dev,e2e]"`,
`playwright install chromium`, then `pytest -m e2e`. Without Playwright or a browser the
tests skip. Locally, `ORIGIN_E2E_CHANNEL=msedge` uses the installed Edge instead.
"""

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

import chromadb  # noqa: E402
import uvicorn  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from main import app  # noqa: E402
from origin.api.routes import health as health_route  # noqa: E402
from origin.api.routes import setup as setup_route  # noqa: E402
from origin.api.routes.chat import get_engine, get_sessions  # noqa: E402
from origin.api.routes.memory import get_curator, get_memory  # noqa: E402
from origin.core import LLMEngine  # noqa: E402
from origin.core.ollama import OllamaStatus  # noqa: E402
from origin.memory import VectorMemory  # noqa: E402
from origin.memory.curator import MemoryCurator  # noqa: E402
from origin.memory.storage import SessionStore  # noqa: E402

pytestmark = pytest.mark.e2e

REPLY = "Olá! Eu sou o Origin, rodando localmente."


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _healthy(settings, timeout: float = 2.0) -> OllamaStatus:
    models = {settings.ollama_model: True, settings.ollama_embed_model: True}
    return OllamaStatus(url=settings.ollama_base_url, reachable=True, models=models, loaded=models)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    memory = VectorMemory(DeterministicFakeEmbedding(size=16), client, "e2e")
    engine = LLMEngine(
        FakeListChatModel(responses=[REPLY]), system_prompt="test", model_name="fake", memory=memory
    )
    sessions = SessionStore(tmp_path_factory.mktemp("e2e") / "sessions.db")
    curator = MemoryCurator(memory, llm=None)
    app.dependency_overrides.update(
        {
            get_engine: lambda: engine,
            get_memory: lambda: memory,
            get_sessions: lambda: sessions,
            get_curator: lambda: curator,
        }
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(health_route, "check_ollama", _healthy)  # the status light: online
    patch.setattr(setup_route, "check_ollama", _healthy)  # no first-run dialog

    port = _free_port()
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uv.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the e2e server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True
    thread.join(10)
    patch.undo()
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    channel = os.environ.get("ORIGIN_E2E_CHANNEL") or None
    with playwright_api.sync_playwright() as pw:
        try:
            instance = pw.chromium.launch(channel=channel)
        except Exception as exc:  # no browser installed (run `playwright install chromium`)
            if os.environ.get("ORIGIN_E2E_REQUIRED"):  # CI: a missing browser is a failure
                raise
            pytest.skip(f"no browser for the e2e tests: {str(exc).splitlines()[0]}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, server: str):
    context = browser.new_context(viewport={"width": 1280, "height": 860}, locale="pt-BR")
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg: msg.type == "error" and errors.append(msg.text))
    page.goto(server)
    page.wait_for_selector("#health.ok")
    page.errors = errors
    yield page
    context.close()
    assert not errors, f"browser errors: {errors}"


def send(page, text: str) -> str:
    # Waits for the reply itself: the fake model answers in milliseconds, so the
    # "pending" bubble can come and go before a wait for it starts.
    before = page.locator(".message.assistant:not(.pending)").count()
    page.fill("#input", text)
    page.click("#send")
    page.wait_for_function(
        """(n) => document.querySelectorAll('.message.assistant:not(.pending)').length > n
               && !document.querySelector('.message.assistant.pending')""",
        arg=before,
    )
    return page.locator(".message.assistant .content").last.inner_text()


# ---------------------------------------------------------------- Markdown

MARKDOWN = json.loads((Path(__file__).parent / "markdown_cases.json").read_text(encoding="utf-8"))


def test_markdown_rendering(page) -> None:
    got = page.evaluate(
        """async (cases) => {
            const { renderMarkdown } = await import('/static/markdown.js');
            return cases.map(([source]) => renderMarkdown(source));
        }""",
        MARKDOWN,
    )
    failures = [
        f"{source!r}\n  got  {html}\n  want {want}"
        for (source, want), html in zip(MARKDOWN, got, strict=True)
        if html != want
    ]
    assert not failures, "\n".join(failures)


def test_markdown_never_breaks_out_of_a_link(page) -> None:
    html = page.evaluate(
        """async () => (await import('/static/markdown.js'))
            .renderMarkdown('https://e.com/"onmouseover="alert(1)')"""
    )
    assert 'onmouseover="' not in html


# ---------------------------------------------------------------- the interface


def test_chat_round_trip_and_the_session_list(page) -> None:
    assert page.title() == "Origin"
    assert send(page, "Oi, tudo bem?") == REPLY
    page.wait_for_selector("#session-list .session-open")
    assert "Oi, tudo bem?" in page.locator("#session-list").inner_text()
    # The meter reports the window after the turn.
    assert page.locator("#context-label").inner_text().endswith("%")


def test_context_panel_opens_from_the_meter(page) -> None:
    send(page, "Oi")
    page.click("#context-meter")
    panel = page.locator("#context-panel")
    panel.wait_for(state="visible")
    assert "6.144" in panel.inner_text()  # the fixed window of the tests (OLLAMA_NUM_CTX)


def test_internet_switch_survives_a_reload(page, server: str) -> None:
    toggle = page.locator("#web-toggle")
    assert toggle.get_attribute("aria-pressed") == "false"
    toggle.click()
    assert toggle.get_attribute("aria-pressed") == "true"
    page.reload()
    page.wait_for_selector("#health.ok")
    assert page.locator("#web-toggle").get_attribute("aria-pressed") == "true"
    page.locator("#web-toggle").click()  # leave it as the other tests expect


def test_memory_manager_add_archive_restore(page) -> None:
    page.click("#open-memory")
    page.wait_for_selector("#memory-dialog[open]")
    page.click("#toggle-add")
    page.fill("#memory-text", "Meu time é o Sport.\nTenho alergia a camarão.")
    page.evaluate("document.querySelector('#add-memory').requestSubmit()")
    rows = page.locator(".memory-row")
    # Without a model the curator stores the text as written: one memory.
    rows.filter(has_text="camarão").wait_for()

    rows.filter(has_text="camarão").locator(".link").nth(1).click()  # archive
    rows.filter(has_text="camarão").wait_for(state="detached")
    page.click("[data-status=archived]")
    archived = page.locator(".memory-row.archived", has_text="camarão")
    archived.wait_for()
    archived.locator(".link").nth(1).click()  # restore
    archived.wait_for(state="detached")
    page.click("[data-status=active]")
    rows.filter(has_text="camarão").wait_for()


def test_language_switch(page, server: str) -> None:
    page.click("#open-settings")
    page.wait_for_selector("#settings-dialog[open]")
    page.select_option("#language", "en")  # saved, then the page reloads in English
    page.wait_for_function("document.documentElement.lang === 'en'")
    page.wait_for_selector("#health.ok")
    assert page.locator("#new-session").inner_text().strip().lower().startswith("+ new")


def test_phone_width_has_no_horizontal_scroll(browser, server: str) -> None:
    context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
    page = context.new_page()
    page.goto(server)
    page.wait_for_selector("#health.ok", state="attached")  # in the closed sidebar
    overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
    context.close()
    assert not overflow
