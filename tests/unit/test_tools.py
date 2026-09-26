from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from origin.config import Settings
from origin.core import LLMEngine
from origin.integrations import ToolContext, ToolRegistry
from origin.integrations.tools import clock
from origin.memory import VectorMemory

FIXED_NOW = datetime(2026, 9, 26, 18, 30, tzinfo=timezone(timedelta(hours=-3)))


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock, "_now", lambda: FIXED_NOW)


def tools_for(memory: VectorMemory | None = None, **settings: object) -> dict:
    return ToolRegistry.discover(ToolContext(Settings(**settings), memory)).tools


def test_discover_loads_builtin_plugins(memory: VectorMemory) -> None:
    assert set(tools_for(memory)) == {"remember", "get_current_datetime", "days_until"}


def test_discover_skips_memory_tools_without_memory() -> None:
    assert "remember" not in tools_for(None)


def test_discover_respects_disabled(memory: VectorMemory) -> None:
    ctx = ToolContext(Settings(), memory)
    assert "remember" not in ToolRegistry.discover(ctx, disabled=["remember"]).tools


def test_registry_rejects_duplicate_names() -> None:
    @tool
    def dup() -> str:
        """Dup."""
        return ""

    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([dup, dup])


async def test_remember_saves_to_memory(memory: VectorMemory) -> None:
    output = await tools_for(memory)["remember"].ainvoke({"fact": "Meu time é o Sport."})
    assert output.startswith("Saved to long-term memory")
    assert memory.count() == 1


@pytest.mark.usefixtures("frozen_clock")
def test_current_datetime_is_localized() -> None:
    output = tools_for(origin_locale="pt-BR")["get_current_datetime"].invoke({})
    assert output == "weekday: sábado\ndate: 2026-09-26\ntime: 18:30\nutc_offset: -0300"


@pytest.mark.usefixtures("frozen_clock")
def test_days_until_next_occurrence() -> None:
    days_until = tools_for()["days_until"]
    assert days_until.invoke({"target_date": "12-25"}).startswith("days: 90\n")
    assert days_until.invoke({"target_date": "01-01"}).startswith("days: 97\n")
    assert days_until.invoke({"target_date": "2026-09-20"}).startswith("days: -6\n")


@pytest.mark.parametrize(
    ("target", "expected"),
    [("09-26", date(2026, 9, 26)), ("09-25", date(2027, 9, 25)), ("2025-01-01", date(2025, 1, 1))],
)
def test_parse_target(target: str, expected: date) -> None:
    assert clock._parse_target(target, date(2026, 9, 26)) == expected


def test_tools_endpoint_lists_engine_tools(
    client: TestClient, fake_engine: LLMEngine, memory: VectorMemory
) -> None:
    fake_engine.tools = tools_for(memory)

    response = client.get("/tools")

    assert response.status_code == 200
    by_name = {t["name"]: t for t in response.json()}
    assert set(by_name) == {"remember", "get_current_datetime", "days_until"}
    assert "target_date" in by_name["days_until"]["args"]
