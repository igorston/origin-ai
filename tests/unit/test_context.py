import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from main import app
from origin.api.routes.chat import get_context, get_engine
from origin.core import ChatTurn, LLMEngine
from origin.core.context import (
    SECTIONS,
    ContextBudget,
    ContextClosed,
    ContextManager,
    ConversationSummarizer,
    MessageTooLong,
    estimate_tokens,
    guard_details,
    join_sections,
    key_details,
    split_sections,
)
from origin.memory.storage import SessionStore
from tests.fakes import ScriptedChatModel


class FakeSummarizer:
    """Deterministic 'summary': the concatenated contents, truncated."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def fold(self, summary: str, turns, max_tokens: int | None = None):
        self.calls.append((summary, len(turns)))
        folded = " / ".join(t.content[:20] for t in turns)
        merged = f"{summary} | {folded}".strip(" |")
        # Folding only grows the summary; condensing (no turns) shrinks it to the budget.
        return (merged if turns else merged[: (max_tokens or 400) * 2]), 0


class FactsSummarizer(FakeSummarizer):
    """Like the real summarizer on a chat full of facts: what the user says lands in USER
    FACTS, which condensing never shrinks (only protect=False does)."""

    async def fold(self, summary, turns, max_tokens=None, *, protect=True):
        self.calls.append((summary, len(turns)))
        parts = split_sections(summary) if summary else dict.fromkeys(SECTIONS, "")
        new = "".join(f"\n- {t.content}" for t in turns if t.role == "user")
        facts = parts["USER FACTS"] + new
        if not protect:
            facts = facts[: (max_tokens or 400) * 2]
        sections = {"USER FACTS": facts.strip(), "TOPICS": "- conversa", "PRESERVED": ""}
        return join_sections(sections), 0


def budget(**overrides) -> ContextBudget:
    values = dict(
        window=600,
        reply_reserve=100,
        compact_at=0.75,
        compact_target=0.5,
        warn_at=0.6,
        critical_at=0.9,
        keep_recent=4,
        min_recent=2,
        summary_max_tokens=120,
        memory_reserve=20,
    )
    return ContextBudget(**{**values, **overrides})


def turns(n: int, size: int = 60) -> list[ChatTurn]:
    return [
        ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"mensagem {i} " + "x" * size)
        for i in range(n)
    ]


def test_estimate_is_conservative() -> None:
    # qwen3 measured 27 tokens for this sentence; the estimate must not be lower.
    assert (
        estimate_tokens(
            "Minha irmã Júlia adora chocolate amargo e mora em Porto Alegre desde 2019."
        )
        >= 27
    )


def test_usage_states() -> None:
    manager = ContextManager(budget(), None, base_tokens=50)
    assert manager.usage("", turns(2)).state == "ok"
    assert manager.usage("", turns(8)).state == "warning"  # ~64% of the usable budget
    assert manager.usage("", turns(12)).state == "critical"
    assert manager.usage("", turns(2), closed=True).state == "closed"
    measured = manager.usage("", turns(2), measured=300)
    assert measured.measured and measured.used == 300 and measured.percent == 0.6


async def test_optimize_folds_old_turns_and_keeps_recent() -> None:
    summarizer = FakeSummarizer()
    manager = ContextManager(budget(compact_target=0.75), summarizer, base_tokens=50)

    result = await manager.optimize("", turns(10), "oi")

    assert result.steps and result.compactions == len(result.steps)
    assert len(result.turns) == 4  # keep_recent
    assert result.summarized == 6 and "mensagem 0" in result.summary
    assert result.compressions == 0


async def test_optimize_folds_down_to_the_target_not_just_below_the_trigger() -> None:
    manager = ContextManager(budget(keep_recent=6), FakeSummarizer(), base_tokens=50)
    result = await manager.optimize("", turns(11), "oi")
    # One fold to 6 kept messages would stop at ~57%; hysteresis goes on to min_recent.
    assert manager.usage(result.summary, result.turns, "oi").percent < 0.5
    assert len(result.turns) == 2 and len(result.steps) == 2


async def test_optimize_leaves_small_conversations_alone() -> None:
    summarizer = FakeSummarizer()
    manager = ContextManager(budget(), summarizer, base_tokens=50)
    result = await manager.optimize("", turns(2), "oi")
    assert (result.summary, len(result.turns), result.steps, summarizer.calls) == ("", 2, [], [])


async def test_long_summaries_are_condensed_and_counted() -> None:
    summarizer = FakeSummarizer()
    manager = ContextManager(budget(summary_max_tokens=20), summarizer, base_tokens=50)
    result = await manager.optimize("", turns(10), "oi")
    assert result.steps[0].compressed is True and result.compressions >= 1
    assert summarizer.calls[1] == (summarizer.calls[1][0], 0)  # a fold with no turns = condense


async def test_there_is_no_fixed_cap_on_condensations() -> None:
    # 50 condensations so far: a chat of general topics can keep condensing them.
    manager = ContextManager(budget(summary_max_tokens=20), FakeSummarizer(), base_tokens=50)
    result = await manager.optimize("", turns(10), "oi", compactions=50, compressions=50)
    assert result.compressions > 50 and estimate_tokens(result.summary) <= 40


async def test_condensing_is_skipped_when_protected_facts_dominate() -> None:
    summarizer = FactsSummarizer()
    manager = ContextManager(budget(summary_max_tokens=20), summarizer, base_tokens=50)
    result = await manager.optimize("", turns(10), "oi")
    # Over budget, but TOPICS are a sliver of it: condensing could not make room.
    assert result.compressions == 0 and all(n > 0 for _, n in summarizer.calls)
    assert manager.usage(result.summary, result.turns).protected > 20


async def test_a_condensation_that_barely_helps_is_not_repeated_in_the_turn() -> None:
    class Stubborn(FakeSummarizer):
        async def fold(self, summary, turns, max_tokens=None):
            self.calls.append((summary, len(turns)))
            if not turns:
                return summary, 0  # "condensed" without shrinking anything
            return f"{summary} | " + " / ".join(t.content[:20] for t in turns), 0

    summarizer = Stubborn()
    manager = ContextManager(budget(summary_max_tokens=20, keep_recent=2), summarizer, 50)
    await manager.optimize("", turns(14), "oi")
    assert sum(1 for _, n in summarizer.calls if n == 0) == 1


async def test_chat_closes_only_when_protected_content_no_longer_fits() -> None:
    facts = "### USER FACTS\n" + "- pedido PX-1037 da cliente Marta vence em 2 de março\n" * 40
    manager = ContextManager(budget(summary_max_tokens=20), FakeSummarizer(), base_tokens=50)
    with pytest.raises(ContextClosed, match="fatos e textos preservados"):
        await manager.optimize(facts, turns(10), "oi")

    # The same size in TOPICS is condensed instead, and the chat goes on.
    topics = "### TOPICS\n" + "- explicação sobre o protocolo HTTPS e seus certificados\n" * 40
    result = await manager.optimize(topics, turns(10), "oi")
    assert result.compressions >= 1 and manager.usage(result.summary, result.turns).percent <= 1


def test_detail_guard_preserves_user_sentences_verbatim() -> None:
    user = (
        "Meu voo para Lisboa é no dia 14 de novembro, o código da reserva é XK7Q2P. "
        "Adoro conversar com você."
    )
    lossy = "### USER FACTS\n- Vai viajar para Lisboa em novembro.\n### TOPICS\n- HTTPS"
    guarded, added = guard_details([user], lossy)
    parts = split_sections(guarded)
    assert added == 1  # the sentence without specific details is not copied
    assert "XK7Q2P" in parts["PRESERVED"] and "14 de novembro" in parts["PRESERVED"]
    assert "HTTPS" in parts["TOPICS"]
    # Idempotent once the details are there.
    assert guard_details([user], guarded) == (guarded, 0)


def test_key_details() -> None:
    details = key_details(
        "Estou refatorando o billing_service.py; a função calcular_fatura tem um bug. "
        "O nome do meu gerente é Ricardo Almeida e o orçamento é de R$ 7.500."
    )
    assert {"billing_service.py", "calcular_fatura", "Ricardo", "Almeida", "7.500"} <= details
    assert "Estou" not in details and "O" not in details


async def test_summarizer_keeps_preserved_section_and_guards_facts() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage("### USER FACTS\n- Tem um gerente.\n### TOPICS\n- nada\n### PRESERVED\n-")
        ]
    )
    summarizer = ConversationSummarizer(model, max_tokens=200)
    previous = "### USER FACTS\n-\n### TOPICS\n-\n### PRESERVED\n- Meu código é AB12."
    summary, added = await summarizer.fold(
        previous, [ChatTurn(role="user", content="O nome do meu gerente é Ricardo Almeida.")]
    )
    parts = split_sections(summary)
    assert "AB12" in parts["PRESERVED"] and "Ricardo Almeida" in parts["PRESERVED"]
    assert added == 1


async def test_message_too_long_is_rejected_without_closing() -> None:
    manager = ContextManager(budget(), FakeSummarizer(), base_tokens=50)
    with pytest.raises(MessageTooLong):
        await manager.optimize("", [], "x" * 5000, 0, 0)


def test_trim_client_history() -> None:
    manager = ContextManager(budget(), None, base_tokens=50)
    kept, dropped = manager.trim(turns(12), "oi")
    assert dropped > 0 and len(kept) == 12 - dropped
    assert manager.usage("", kept, "oi").percent < 0.75


def test_session_store_migrates_old_databases(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
            " role TEXT NOT NULL, content TEXT NOT NULL, tool_calls TEXT NOT NULL DEFAULT '[]',"
            " created_at TEXT NOT NULL);"
            "INSERT INTO sessions VALUES ('s1', 'Antiga', '2026-01-01', '2026-01-01');"
            "INSERT INTO messages (session_id, role, content, created_at)"
            " VALUES ('s1', 'user', 'oi', '2026-01-01');"
        )
    store = SessionStore(path)
    session = store._get("s1")
    assert (session.title, session.status, session.summary, session.compactions) == (
        "Antiga",
        "open",
        "",
        0,
    )
    assert store._messages("s1", None, 0)[0].tokens == 0


@pytest.fixture
def small_context() -> ContextManager:
    manager = ContextManager(
        budget(window=700, summary_max_tokens=60), FakeSummarizer(), base_tokens=50
    )
    app.dependency_overrides[get_context] = lambda: manager
    return manager


def test_session_is_optimized_then_closed_then_continued(client: TestClient, small_context) -> None:
    # A chat full of facts: they are never condensed, so they eventually fill the window.
    small_context.summarizer = FactsSummarizer()
    model = ScriptedChatModel(responses=[AIMessage("resposta " + "y" * 120)])
    app.dependency_overrides[get_engine] = lambda: LLMEngine(model, "sys", "scripted")
    session_id = client.post("/sessions").json()["id"]

    compactions = []
    for i in range(40):
        response = client.post(
            "/chat", json={"message": f"pergunta {i} " + "z" * 80, "session_id": session_id}
        )
        if response.status_code == 409:
            break
        assert response.status_code == 200
        compactions += response.json()["compactions"]
    else:
        pytest.fail("the chat never reached its operational limit")

    # Optimized several times before closing, without condensing facts that cannot shrink.
    assert len(compactions) > 3
    assert sum(c["compressed"] for c in compactions) == 0
    closed = response.json()["detail"]
    assert closed["closed"] is True and "fatos e textos preservados" in closed["reason"]
    assert closed["context"]["protected"] > closed["context"]["protected_room"] / 2
    # The model saw the running summary in its system prompt once it existed.
    assert any("Earlier in this conversation" in call[0].content for call in model.received)

    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["status"] == "closed"
    assert detail["summary"] and detail["summarized_upto"] > 0
    assert detail["context"]["state"] == "closed"
    assert "pergunta" in detail["summary"]  # the work of the closing turn is kept too
    assert len(detail["messages"]) >= 6  # the full transcript is kept for the user
    # Further messages are refused, not silently dropped.
    assert client.post("/chat", json={"message": "oi", "session_id": session_id}).status_code == 409

    successor = client.post(f"/sessions/{session_id}/continue")
    assert successor.status_code == 201
    new = successor.json()
    assert new["parent_id"] == session_id and new["summary"].startswith(detail["summary"][:10])
    assert new["title"].startswith("↪ ")
    assert client.post("/chat", json={"message": "oi", "session_id": new["id"]}).status_code == 200


def test_oversized_message_is_413_and_chat_stays_open(client: TestClient, small_context) -> None:
    session_id = client.post("/sessions").json()["id"]
    response = client.post("/chat", json={"message": "x" * 5000, "session_id": session_id})
    assert response.status_code == 413
    assert client.get(f"/sessions/{session_id}").json()["status"] == "open"


def test_events_announce_optimizations(client: TestClient, small_context) -> None:
    model = ScriptedChatModel(responses=[AIMessage("resposta " + "y" * 120)])
    app.dependency_overrides[get_engine] = lambda: LLMEngine(model, "sys", "scripted")
    session_id = client.post("/sessions").json()["id"]
    for i in range(6):
        client.post("/chat", json={"message": f"p{i} " + "z" * 80, "session_id": session_id})

    response = client.post(
        "/chat/events", json={"message": "mais uma " + "z" * 80, "session_id": session_id}
    )

    first = response.text.split("\n\n")[0]
    assert first.startswith("event: context")
    assert '"compactions": [{' in first


async def test_measurements_calibrate_the_estimate() -> None:
    manager = ContextManager(budget(), FakeSummarizer(), base_tokens=50)
    history = turns(10)  # estimated past compact_at
    estimated = manager.usage("", history).history
    # Ollama measured the same text at 60% of our estimate: no need to compact yet.
    scale = manager.calibrate("", history, 50 + int(estimated * 0.6))
    assert 0.59 < scale < 0.61
    assert manager.usage("", history, "oi").percent >= 0.75
    result = await manager.optimize("", history, "oi", measured=50 + int(estimated * 0.6))
    assert result.steps == [] and result.scale == scale
    # Without a measurement (or a nonsense one) the conservative estimate is kept.
    assert manager.calibrate("", history, None) == 1.0
    assert manager.calibrate("", history, 10) == 0.5
    assert manager.calibrate("", history, 10_000) == 2.0  # dense text: capped at 2x


def test_translated_headers_are_understood() -> None:
    parts = split_sections(
        "### FATOS DO USUÁRIO\n- Mora em Recife.\n### Tópicos\n- HTTPS\n### PRESERVADO\n- x"
    )
    assert parts == {"USER FACTS": "- Mora em Recife.", "TOPICS": "- HTTPS", "PRESERVED": "- x"}
    # Without any header, everything counts as topics (condensable), nothing is lost.
    assert split_sections("texto livre")["TOPICS"] == "texto livre"


def test_window_check() -> None:
    assert ContextManager(budget(), None, base_tokens=50).check_window() is None
    problem = ContextManager(budget(), None, base_tokens=300).check_window()
    assert problem and "too small" in problem


class StubbornSummarizer(FakeSummarizer):
    """Keeps protected facts when condensing, like the real prompt: only protect=False
    shrinks the summary."""

    async def fold(self, summary, turns, max_tokens=None, *, protect=True):
        self.calls.append((summary, len(turns), protect))
        if turns:
            return summary + " " + " / ".join(t.content for t in turns), 0
        return (summary if protect else summary[: (max_tokens or 400) * 2]), 0


async def test_carry_over_fits_the_budget_even_if_facts_must_shrink() -> None:
    summarizer = StubbornSummarizer()
    manager = ContextManager(budget(summary_max_tokens=60), summarizer, base_tokens=50)
    summary = await manager.carry_over("fato " * 100, turns(4))
    assert estimate_tokens(summary) <= 60
    assert [call[2] for call in summarizer.calls] == [True, True, False]  # protected first

    small = await ContextManager(budget(), StubbornSummarizer(), 50).carry_over("oi", turns(1))
    assert small.startswith("oi")  # nothing to condense


async def test_a_huge_last_message_is_folded_instead_of_closing_the_chat() -> None:
    manager = ContextManager(budget(), FakeSummarizer(), base_tokens=50)
    story = [
        ChatTurn(role="user", content="Crie uma história longa"),
        ChatTurn(role="assistant", content="# Título\n\n" + "Era uma vez. " * 200),  # > window
    ]
    result = await manager.optimize("", story, "gostei!")
    assert result.turns == [] and "Crie uma história" in result.summary
    assert manager.usage(result.summary, result.turns, "gostei!").percent <= 1


def test_summarizer_input_keeps_the_start_and_end_of_long_messages() -> None:
    from origin.core.context import MAX_TURN_CHARS, _shorten

    text = "início " + "x" * 10_000 + " fim"
    short = _shorten(text)
    assert len(short) < MAX_TURN_CHARS + 20 and short.startswith("início") and short.endswith("fim")
    assert _shorten("curto") == "curto"


def test_shortened_stories_keep_their_chapter_headings() -> None:
    from origin.core.context import _shorten

    chapters = "".join(f"## Capítulo {i}\n\n" + "texto " * 300 + "\n\n" for i in range(1, 6))
    short = _shorten("# A História\n\n" + chapters)
    assert all(f"## Capítulo {i}" in short for i in range(1, 6))


def test_written_pieces_are_recorded_verbatim_in_the_summary() -> None:
    from origin.core.context import piece_outline

    story = (
        "# O Pacto da Sombra\n\n## O Último Conselho\n\nTexto.\n\n"
        "## A Coroa de Fogo\n\nE assim terminou."
    )
    line = piece_outline(story)
    assert '"O Pacto da Sombra"' in line and "O Último Conselho; A Coroa de Fogo" in line
    assert line.endswith('E assim terminou."')
    assert piece_outline("## Parte 2\n\nContinua.").startswith("- Continuation")
    assert piece_outline("Uma resposta normal.") is None


async def test_summarizer_records_pieces_in_preserved() -> None:
    model = ScriptedChatModel(
        responses=[AIMessage("### USER FACTS\n-\n### TOPICS\n- Uma história\n### PRESERVED\n-")]
    )
    summary, _ = await ConversationSummarizer(model, max_tokens=200).fold(
        "",
        [
            ChatTurn(role="user", content="Crie uma história"),
            ChatTurn(role="assistant", content="# O Farol\n\n## A Noite\n\nFim."),
        ],
    )
    assert '"O Farol"; chapters: A Noite' in split_sections(summary)["PRESERVED"]


def test_counted_replies_are_not_rescaled_by_a_noisy_calibration() -> None:
    from origin.api.routes.chat import CountedTurn

    manager = ContextManager(budget(window=8000), None, base_tokens=1300)
    greeting = [ChatTurn(role="user", content="Oi!"), ChatTurn(role="assistant", content="Olá!")]
    # The greeting's measurement included ~160 tokens of recalled memories: a tiny
    # sample like this must not calibrate anything.
    scale = manager.calibrate("", greeting, measured=1300 + 180)
    assert scale == 1.0
    story = [
        *greeting,
        CountedTurn("user", "Crie uma história", 10),
        CountedTurn("assistant", "x", 3455),
    ]
    usage = manager.usage("", story, scale=2.0)  # even with a skewed scale
    assert 3455 <= usage.history < 3455 + 50  # the story's own count is kept as is
