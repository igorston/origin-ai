from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from origin.config import Settings
from origin.core import LLMEngine
from origin.integrations import ToolContext, ToolRegistry
from origin.integrations.tools import clock
from origin.memory import VectorMemory
from origin.memory.curator import MemoryCurator
from tests.fakes import ScriptedChatModel

FIXED_NOW = datetime(2026, 9, 26, 18, 30, tzinfo=timezone(timedelta(hours=-3)))


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock, "_now", lambda: FIXED_NOW)


def tools_for(memory: VectorMemory | None = None, llm: object = None, **settings: object) -> dict:
    return ToolRegistry.discover(ToolContext(Settings(**settings), memory, llm=llm)).tools


BUILTIN_TOOLS = {
    "remember",
    "forget",
    "get_current_datetime",
    "days_until",
    "date_offset",
    "web_search",
    "fetch_url",
}


def test_discover_loads_builtin_plugins(memory: VectorMemory) -> None:
    assert set(tools_for(memory)) == BUILTIN_TOOLS


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


async def test_remember_does_not_store_duplicates(memory: VectorMemory) -> None:
    await memory.add(["Meu time é o Sport."])
    output = await tools_for(memory)["remember"].ainvoke({"fact": "Meu time é o Sport."})
    assert output.startswith("Already in memory")
    assert memory.count() == 1


@pytest.mark.parametrize(("judge_says", "archived"), [("NO", True), ("YES", False)])
async def test_remember_archives_superseded_memories(
    memory: VectorMemory, judge_says: str, archived: bool
) -> None:
    (old_id,) = await memory.add(["Moro em Recife."])
    judge = FakeListChatModel(responses=[judge_says])
    # Fake embeddings are random, so treat every stored memory as similar.
    tools = tools_for(memory, llm=judge, memory_conflict_threshold=-1.0)

    output = await tools["remember"].ainvoke({"fact": "Moro em São Paulo."})

    assert output.startswith("Saved to long-term memory")
    assert ("It replaces" in output) is archived
    assert bool(memory.get(old_id).metadata.get("archived")) is archived
    assert memory.count() == 2


async def test_remember_keeps_memories_when_judge_unavailable(memory: VectorMemory) -> None:
    (old_id,) = await memory.add(["Moro em Recife."])
    tools = tools_for(memory, llm=None, memory_conflict_threshold=-1.0)

    await tools["remember"].ainvoke({"fact": "Moro em São Paulo."})

    assert not memory.get(old_id).metadata.get("archived")


async def test_forget_by_id_and_by_description(memory: VectorMemory) -> None:
    first, second = await memory.add(["Tenho alergia a camarão.", "Moro em Recife."])
    forget = tools_for(memory)["forget"]

    assert "Deleted memory" in await forget.ainvoke({"memory": first})
    assert "No memory with id" in await forget.ainvoke({"memory": first})
    assert "Deleted memory" in await forget.ainvoke({"memory": "Moro em Recife."})
    assert "nothing was deleted" in await forget.ainvoke({"memory": "xyz"})
    assert memory.count() == 0


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


@pytest.mark.usefixtures("frozen_clock")
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"years": 1}, "date: 2027-09-26 (domingo)\n"),
        ({"days": -1}, "date: 2026-09-25 (sexta-feira)\n"),
        ({"weeks": 2}, "date: 2026-10-10 (sábado)\n"),
        ({"months": -3}, "date: 2026-06-26 (sexta-feira)\n"),
        ({}, "date: 2026-09-26 (sábado)\n"),
    ],
)
def test_date_offset(args: dict, expected: str) -> None:
    output = tools_for(origin_locale="pt-BR")["date_offset"].invoke(args)
    assert output.startswith(expected)
    assert "today: 2026-09-26 (sábado)" in output


@pytest.mark.parametrize(
    ("start", "kwargs", "expected"),
    [
        (date(2026, 1, 31), {"months": 1}, date(2026, 2, 28)),  # clamps to month end
        (date(2024, 2, 29), {"years": 1}, date(2025, 2, 28)),  # leap day
        (date(2026, 12, 15), {"months": 1}, date(2027, 1, 15)),  # year rollover
        (date(2026, 3, 15), {"months": -3}, date(2025, 12, 15)),
        (date(2026, 9, 27), {"years": 1, "days": 1}, date(2027, 9, 28)),
    ],
)
def test_shift_date(start: date, kwargs: dict, expected: date) -> None:
    assert clock.shift_date(start, **kwargs) == expected


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
    assert set(by_name) == BUILTIN_TOOLS
    assert "target_date" in by_name["days_until"]["args"]


@pytest.mark.parametrize(
    ("reply", "replaces"),
    [
        ("NO", True),
        ("A: football team | B: football team | NO", True),
        ("A: son's name | B: daughter's name | YES", False),
        ("I am not sure", False),
    ],
)
async def test_conflict_verdict_is_the_last_answer(reply: str, replaces: bool) -> None:
    curator = MemoryCurator(None, ScriptedChatModel(responses=[AIMessage(reply)]))
    assert (
        await curator.supersedes("Meu time favorito é o Sport.", "Meu time é o Náutico.")
        is replaces
    )


async def test_an_addition_never_replaces() -> None:
    judge = ScriptedChatModel(responses=[AIMessage("NO")])
    curator = MemoryCurator(None, judge)
    assert not await curator.supersedes(
        "Gosto de pizza de calabresa.", "Agora também gosto de pizza de mussarela."
    )
    assert not judge.received  # decided without asking the model
