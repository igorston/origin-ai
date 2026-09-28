# ruff: noqa: E501  (search-result HTML fixtures)
import socket

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from origin.config import Settings
from origin.core import LLMEngine
from origin.core.agent import ToolCallRecord
from origin.integrations import ToolContext, ToolRegistry
from origin.integrations import web as web_module
from origin.integrations.web import (
    WebClient,
    WebError,
    check_public,
    html_to_text,
    needs_live_data,
    parse_duckduckgo,
    sources_note,
)
from tests.fakes import ScriptedChatModel

ADDRESSES = {
    "example.com": "93.184.216.34",
    "evil.example": "93.184.216.35",
    "internal.example": "10.0.0.5",
    "localhost": "127.0.0.1",
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch) -> None:
    def resolve(host, port, *args, **kwargs):
        try:
            address = ADDRESSES.get(host) or str(__import__("ipaddress").ip_address(host))
        except ValueError as exc:
            raise socket.gaierror(host) from exc
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(web_module.socket, "getaddrinfo", resolve)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/tags",  # the local Ollama
        "http://localhost/admin",
        "http://192.168.0.1/",  # the router
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://internal.example/",  # a public name pointing inside
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
    ],
)
async def test_private_and_non_http_addresses_are_refused(url: str) -> None:
    with pytest.raises(WebError):
        await check_public(url)


async def test_public_addresses_and_the_intranet_switch() -> None:
    await check_public("https://example.com/page")
    await check_public("http://192.168.0.1/", allow_private=True)


def client_for(handler) -> WebClient:
    return WebClient(Settings(), transport=httpx.MockTransport(handler))


async def test_fetch_reads_text_and_skips_the_noise() -> None:
    html = (
        "<html><head><title>Notícia</title><script>track()</script></head><body>"
        "<nav>Menu Home</nav><h1>Chuva em Recife</h1><p>Previsão de chuva forte.</p>"
        "<footer>© site</footer></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=html)

    url, title, text = await client_for(handler).fetch("https://example.com/n")
    assert (url, title) == ("https://example.com/n", "Notícia")
    assert "Chuva em Recife" in text and "Previsão de chuva forte." in text
    assert "track()" not in text and "Menu Home" not in text and "© site" not in text


async def test_redirects_are_checked_at_every_hop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "evil.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
        return httpx.Response(200, text="should never be reached")

    with pytest.raises(WebError, match="private or local"):
        await client_for(handler).fetch("https://evil.example/go")


async def test_binary_content_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    with pytest.raises(WebError, match="not a text page"):
        await client_for(handler).fetch("https://example.com/image")


DDG = """
<div class="result results_links results_links_deep result--ad"><div class="links_main">
  <a class="result__a" href="https://ads.example/buy">Compre já</a>
  <a class="result__snippet">Oferta</a></div></div>
<div class="result results_links results_links_deep web-result"><div class="links_main">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://www.python.org/downloads/">Download &amp; Python</a></h2>
  <a class="result__snippet" href="#">The latest version is <b>Python 3.14</b>.</a></div></div>
<div class="result results_links web-result"><div class="links_main">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=x">Docs</a>
  <a class="result__snippet">Documentation</a></div></div>
"""


def test_duckduckgo_results_without_ads() -> None:
    results = parse_duckduckgo(DDG)
    assert [r.url for r in results] == [
        "https://www.python.org/downloads/",
        "https://docs.python.org/3/",
    ]
    assert results[0].title == "Download & Python"
    assert results[0].snippet == "The latest version is Python 3.14."


def test_html_to_text_keeps_paragraphs() -> None:
    title, text = html_to_text("<title> T </title><p>Um</p><p>Dois</p>")
    assert title == "T" and text == "Um\n\nDois"


SEARCH_OUTPUT = (
    "[Untrusted web content...]\nResults for 'x':\n\n"
    "1. Python\n   https://www.python.org/downloads/\n   latest\n"
    "2. Docs\n   https://docs.python.org/3/\n   docs\n"
)


def test_sources_are_added_when_the_answer_names_no_url() -> None:
    records = [ToolCallRecord(name="web_search", args={}, output=SEARCH_OUTPUT)]
    note = sources_note("A versão mais recente é a 3.14 [2].", records, "Brazilian Portuguese")
    assert note == "\n\n**Fontes:**\n1. https://docs.python.org/3/"
    top = sources_note("É a 3.14.", records, "English")  # no [n]: the top results
    assert "**Sources:**" in top and "python.org/downloads" in top and "docs.python.org" in top
    assert sources_note("Veja https://x.org", records, "English") is None  # already cited
    assert (
        sources_note("Oi", [ToolCallRecord(name="remember", args={}, output="ok")], "English")
        is None
    )
    page = [
        ToolCallRecord(
            name="fetch_url", args={}, output="[...]\nURL: https://example.com/a\nTitle: A"
        )
    ]
    assert "https://example.com/a" in sources_note("Resumo.", page, "English")


@pytest.mark.parametrize(
    ("message", "live"),
    [
        ("Qual é a cotação do dólar hoje?", True),
        ("Vai chover amanhã em Recife?", True),
        ("Quem ganhou o jogo do Flamengo ontem?", True),
        ("What's the weather in Lisbon?", True),
        ("Me explique a MP das Bets?", True),
        ("O que o STF decidiu sobre o marco temporal?", True),
        ("Como fica a reforma tributária para autônomos?", True),
        ("Explique o que é recursão em uma frase.", False),
        ("Me explique o que é uma função lambda", False),
        ("Qual a capital da França?", False),
        ("Que dia é hoje?", False),
        ("Explique o que é uma API REST.", False),
    ],
)
def test_live_data_detection(message: str, live: bool) -> None:
    assert needs_live_data(message) is live


# ---------------------------------------------------------------- the engine


def web_tools(outputs: list[str]):
    @tool
    async def web_search(query: str) -> str:
        """Search the web."""
        outputs.append(query)
        return SEARCH_OUTPUT

    @tool
    def get_current_datetime() -> str:
        """Current date."""
        return "date: 2026-09-28"

    web_search.metadata = {"network": True}
    return {"web_search": web_search, "get_current_datetime": get_current_datetime}


async def test_network_tools_are_offered_only_when_asked() -> None:
    model = ScriptedChatModel(responses=[AIMessage("Oi!")])
    engine = LLMEngine(model, "sys", "scripted", tools=web_tools([]))
    await engine.generate("Oi", use_memory=False)
    assert model.bound_tools == ["get_current_datetime"]
    await engine.generate("Oi", use_memory=False, use_web=True)
    assert model.bound_tools == ["web_search", "get_current_datetime"]


async def test_live_data_always_gets_a_search_and_sources() -> None:
    searched: list[str] = []
    router = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[{"name": "get_current_datetime", "args": {}, "id": "1"}])
        ]
    )
    model = ScriptedChatModel(responses=[AIMessage("O dólar está a R$ 5,22 [1].")])
    engine = LLMEngine(
        model, "sys", "scripted", tools=web_tools(searched), tool_routing=True, router=router
    )

    result = await engine.generate(
        "Qual é a cotação do dólar hoje?", use_memory=False, use_web=True
    )

    assert searched == ["Qual é a cotação do dólar hoje?"]  # added although the router skipped it
    assert result.text.endswith("**Fontes:**\n1. https://www.python.org/downloads/")


def test_web_access_off_removes_the_tools() -> None:
    names = set(ToolRegistry.discover(ToolContext(Settings(web_access=False))).tools)
    assert not names & {"web_search", "fetch_url"}
    assert {"web_search", "fetch_url"} <= set(ToolRegistry.discover(ToolContext(Settings())).tools)


async def test_without_internet_live_data_is_not_invented() -> None:
    model = ScriptedChatModel(responses=[AIMessage("Não consigo consultar agora.")])
    engine = LLMEngine(model, "sys", "scripted", tools=web_tools([]))
    await engine.generate("Qual é a cotação do dólar hoje?", use_memory=False)  # 🌐 off
    prompt = model.received[-1][-1].content
    assert "no internet access" in prompt and "🌐" in prompt
    await engine.generate("Qual a capital da França?", use_memory=False)
    assert "no internet access" not in model.received[-1][-1].content


@pytest.mark.parametrize(
    ("message", "deep"),
    [
        ("Me explique a MP das Bets?", True),
        ("O que muda com a reforma tributária?", True),
        ("Explain the new law", True),
        ("Qual é a cotação do dólar hoje?", False),
        ("Quem ganhou o jogo?", False),
    ],
)
def test_explanation_requests(message: str, deep: bool) -> None:
    assert web_module.wants_depth(message) is deep


async def test_explanations_read_the_top_result_in_the_users_language() -> None:
    pages: list[str] = []

    @tool
    async def web_search(query: str) -> str:
        """Search the web."""
        return SEARCH_OUTPUT

    @tool
    async def fetch_url(url: str) -> str:
        """Read a page."""
        pages.append(url)
        if "python.org/downloads" in url:  # the top result blocks robots (403)
            return f"Could not read {url}: {url} answered HTTP 403"
        return f"[...]\nURL: {url}\nTitle: MP\n\nTexto completo da página."

    for t in (web_search, fetch_url):
        t.metadata = {"network": True}
    router = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "MP"}, "id": "1"}])
        ]
    )
    model = ScriptedChatModel(responses=[AIMessage("A MP proíbe as apostas de quota fixa.")])
    engine = LLMEngine(
        model,
        "sys",
        "scripted",
        tools={"web_search": web_search, "fetch_url": fetch_url},
        tool_routing=True,
        router=router,
        locale="pt-BR",
    )

    await engine.generate("Me explique a MP das Bets?", use_memory=False, use_web=True)

    # The top result refused (403): the next one was read instead.
    assert pages == ["https://www.python.org/downloads/", "https://docs.python.org/3/"]
    final_prompt = model.received[-1]
    reminder = final_prompt[-1].content
    assert "português do Brasil" in reminder and "uma ou duas frases" not in reminder


async def test_actions_keep_the_short_reply_instruction() -> None:
    router = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[{"name": "get_current_datetime", "args": {}, "id": "1"}])
        ]
    )
    model = ScriptedChatModel(responses=[AIMessage("Hoje é segunda-feira.")])
    engine = LLMEngine(
        model,
        "sys",
        "scripted",
        tools=web_tools([]),
        tool_routing=True,
        router=router,
        locale="pt-BR",
    )
    await engine.generate("Que dia é hoje?", use_memory=False, use_web=True)
    assert "uma ou duas frases diretas" in model.received[-1][-1].content
