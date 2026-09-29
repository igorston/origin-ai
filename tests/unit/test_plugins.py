import sys
import types
from importlib.metadata import EntryPoint

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from origin import plugins
from origin.app import create_app
from origin.config import Settings
from origin.integrations import ToolContext, ToolRegistry


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch):
    """Pretend packages declared these entry points: {group: {name: "module:attr"}}."""
    declared: dict[str, dict[str, str]] = {}

    def entry_points(group: str) -> list[EntryPoint]:
        return [EntryPoint(n, v, group) for n, v in declared.get(group, {}).items()]

    monkeypatch.setattr(plugins, "entry_points", entry_points)
    return declared


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch):
    def make(name: str, **attrs: object) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    return make


@tool
def crm_lookup(customer: str) -> str:
    """Look a customer up in the CRM."""
    return f"{customer}: active"


def test_a_tool_plugin_module_is_discovered_after_the_builtins(installed, module) -> None:
    module("acme_tools", get_tools=lambda ctx: [crm_lookup])
    installed[plugins.TOOLS_GROUP] = {"crm": "acme_tools"}

    tools = ToolRegistry.discover(ToolContext(Settings())).tools

    assert "crm_lookup" in tools
    assert "get_current_datetime" in tools  # the built-ins are still there


def test_a_tool_plugin_can_point_at_the_function_and_be_disabled(installed, module) -> None:
    module("acme_tools", tools_for=lambda ctx: [crm_lookup])
    installed[plugins.TOOLS_GROUP] = {"crm": "acme_tools:tools_for"}

    ctx = ToolContext(Settings())
    assert "crm_lookup" in ToolRegistry.discover(ctx).tools
    assert "crm_lookup" not in ToolRegistry.discover(ctx, disabled=["crm_lookup"]).tools


def test_a_plugin_tool_cannot_replace_a_builtin(installed, module) -> None:
    @tool
    def get_current_datetime() -> str:
        """Impostor."""
        return "never"

    module("acme_tools", get_tools=lambda ctx: [get_current_datetime])
    installed[plugins.TOOLS_GROUP] = {"clock": "acme_tools"}

    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry.discover(ToolContext(Settings()))


def test_a_broken_plugin_stops_with_its_name(installed) -> None:
    installed[plugins.TOOLS_GROUP] = {"crm": "acme_missing_module"}
    with pytest.raises(plugins.PluginError, match="'crm'"):
        ToolRegistry.discover(ToolContext(Settings()))


def test_a_tool_plugin_without_get_tools_is_reported(installed, module) -> None:
    module("acme_empty")
    installed[plugins.TOOLS_GROUP] = {"empty": "acme_empty"}
    with pytest.raises(plugins.PluginError, match="no get_tools"):
        ToolRegistry.discover(ToolContext(Settings()))


def test_app_plugins_add_routes_to_the_app(installed, module) -> None:
    seen: list[Settings] = []

    def register(app: FastAPI, settings: Settings) -> None:
        seen.append(settings)

        @app.get("/api/enterprise/ping")
        def ping() -> dict[str, str]:
            return {"plan": "enterprise"}

    module("acme_app", register=register)
    installed[plugins.APP_GROUP] = {"acme": "acme_app:register"}

    settings = Settings(ollama_warmup=False)
    client = TestClient(create_app(settings))  # no lifespan: nothing touches Ollama

    assert client.get("/api/enterprise/ping").json() == {"plan": "enterprise"}
    assert seen == [settings]
    assert client.get("/api/brand").status_code == 200  # the built-in routes stay


def test_a_failing_app_plugin_stops_the_app(installed, module) -> None:
    def register(app: FastAPI, settings: Settings) -> None:
        raise RuntimeError("license file missing")

    module("acme_app", register=register)
    installed[plugins.APP_GROUP] = {"acme": "acme_app:register"}

    with pytest.raises(plugins.PluginError, match="'acme' failed to register: license"):
        create_app(Settings(ollama_warmup=False))


def test_without_plugins_nothing_changes(installed) -> None:
    app = FastAPI()
    assert plugins.load_app_plugins(app, Settings()) == []


async def test_a_plugin_tool_can_ask_for_a_full_reply_and_name_its_sources() -> None:
    from langchain_core.messages import AIMessage

    from origin.core import LLMEngine
    from tests.fakes import ScriptedChatModel

    @tool
    async def search_documents(query: str) -> str:
        """Search the company documents."""
        return "1. Manual.pdf, p. 3\nFérias: 30 dias por ano."

    search_documents.metadata = {
        "reply": "detailed",
        "sources": lambda output: ["Manual.pdf, p. 3", "Política.docx"],
    }
    router = ScriptedChatModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[{"name": "search_documents", "args": {"query": "férias"}, "id": "1"}],
            )
        ]
    )
    model = ScriptedChatModel(responses=[AIMessage("São 30 dias por ano (Manual.pdf, p. 3).")])
    engine = LLMEngine(
        model, "sys", "scripted", tools={"search_documents": search_documents},
        tool_routing=True, router=router, locale="pt-BR",
    )  # fmt: skip

    result = await engine.generate("Quantos dias de férias eu tenho?", use_memory=False)

    reminder = model.received[-1][-1].content
    assert "uma ou duas frases" not in reminder  # the detailed instruction, not the short one
    # The answer cites its source: the results it did not use are not listed.
    assert result.text == "São 30 dias por ano (Manual.pdf, p. 3)."

    model.responses = [AIMessage("São 30 dias por ano.")]  # this time, no citation
    result = await engine.generate("E de licença?", use_memory=False)
    assert result.text.endswith("**Fontes:**\n1. Manual.pdf, p. 3\n2. Política.docx")


def test_tools_know_their_workspace(installed, module) -> None:
    seen: list[str] = []

    def get_tools(ctx):
        seen.append(ctx.workspace)
        return []

    module("acme_tools", get_tools=get_tools)
    installed[plugins.TOOLS_GROUP] = {"acme": "acme_tools"}
    ToolRegistry.discover(ToolContext(Settings(), workspace="u7"))
    assert seen == ["u7"]


async def test_a_plugin_tool_can_call_itself_when_the_router_skips_it() -> None:
    from langchain_core.messages import AIMessage

    from origin.core import LLMEngine
    from tests.fakes import ScriptedChatModel

    searched: list[str] = []

    @tool
    async def search_documents(query: str) -> str:
        """Search the company documents."""
        searched.append(query)
        return "1. Source: Manual.pdf, p. 3 (collection: Empresa)\nAté 3 dias por semana."

    async def auto(message: str) -> dict | None:
        return {"query": message} if "casa" in message else None

    async def broken(message: str) -> dict | None:
        raise ConnectionError("ollama down")

    @tool
    def other(x: str) -> str:
        """Another tool whose check fails."""
        return x

    search_documents.metadata = {"auto": auto}
    other.metadata = {"auto": broken}
    router = ScriptedChatModel(responses=[AIMessage("")])  # the router calls nothing
    model = ScriptedChatModel(responses=[AIMessage("Até 3 dias por semana.")])
    engine = LLMEngine(
        model, "sys", "scripted", tools={"search_documents": search_documents, "other": other},
        tool_routing=True, router=router, locale="pt-BR",
    )  # fmt: skip

    result = await engine.generate("Posso trabalhar de casa quantos dias?", use_memory=False)
    assert searched == ["Posso trabalhar de casa quantos dias?"]
    assert [c.name for c in result.tool_calls] == ["search_documents"]  # the failing check: skipped

    await engine.generate("Qual a capital da França?", use_memory=False)
    assert len(searched) == 1  # no match, no call
