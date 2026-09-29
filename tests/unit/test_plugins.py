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
