from datetime import date

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from main import app
from origin.api.routes.memory import get_curator
from origin.memory import VectorMemory
from origin.memory.curator import (
    MemoryCurator,
    added_words,
    missing_words,
    parse_facts,
    resolve_relative_days,
)


def scripted(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


@pytest.mark.parametrize(
    ("output", "facts"),
    [
        ("Moro em Recife.\nTrabalho na Globant.", ["Moro em Recife.", "Trabalho na Globant."]),
        ("- Moro em Recife.\n2. Uso Linux.\n\n", ["Moro em Recife.", "Uso Linux."]),
        ('Facts:\n"Meu time é o Sport."', ["Meu time é o Sport."]),
    ],
)
def test_parse_facts(output: str, facts: list[str]) -> None:
    assert parse_facts(output) == facts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reunião com o time amanhã 14h", "reunião com o time em 28/09/2026 14h"),
        ("Hoje comecei a estudar japonês", "em 27/09/2026 comecei a estudar japonês"),
        ("dentista depois de amanhã", "dentista em 29/09/2026"),
        ("ontem fui ao médico", "em 26/09/2026 fui ao médico"),
        ("tomorrow I fly to Madrid", "on 2026-09-28 I fly to Madrid"),
        ("mañana tengo examen", "el 28/09/2026 tengo examen"),
        ("corro por la mañana", "corro por la mañana"),  # "in the morning"
        ("tenho dentista quinta 15h", "tenho dentista quinta 15h"),  # weekdays stay
        ("hojeando", "hojeando"),  # whole words only
    ],
)
def test_resolve_relative_days(text: str, expected: str) -> None:
    assert resolve_relative_days(text, today=date(2026, 9, 27)) == expected


def test_missing_words_detects_dropped_details_and_translation() -> None:
    note = "Mudei de emprego, agora trabalho na Globant como dev sênior"
    assert missing_words(note, ["Trabalho na Globant."]) == {"dev", "senior"}
    assert missing_words(note, ["Trabalho na Globant como dev sênior."]) == set()
    # Spacing changes and accents are not losses.
    assert missing_words("uso vscode", ["Uso o VS Code."]) == set()
    assert missing_words("sou alergico a camarao", ["Sou alérgico a camarão."]) == set()
    # Translation loses the note's words (names like Lisbon/Lisboa may still match).
    assert "live" in missing_words("I live in Lisbon", ["Moro em Lisboa."])


def test_added_words_detects_invented_facts() -> None:
    note = "My sister Julia loves dark chocolate and lives in Porto"
    invented = added_words(note, ["My sister Julia loves dark chocolate.", "Ana lives in Porto."])
    assert invented == {"ana"}
    assert added_words("meu cachorro thor", ["Meu cachorro se chama Thor."]) == {"chama"}


async def test_normalize_accepts_a_faithful_rewrite(memory: VectorMemory) -> None:
    curator = MemoryCurator(
        memory, scripted("Meu cachorro se chama Thor.\nMinha gata se chama Luna.")
    )
    facts = await curator.normalize("meu cachorro thor e minha gata luna")
    assert facts == ["Meu cachorro se chama Thor.", "Minha gata se chama Luna."]


async def test_normalize_retries_then_falls_back_to_the_original(memory: VectorMemory) -> None:
    note = "trabalho na Globant como dev sênior"
    retry_ok = MemoryCurator(
        memory, scripted("Trabalho na Globant.", "Trabalho na Globant como dev sênior.")
    )
    assert await retry_ok.normalize(note) == ["Trabalho na Globant como dev sênior."]

    always_bad = MemoryCurator(memory, scripted("Trabalho na Globant."))
    assert await always_bad.normalize(note) == [note]

    invents = MemoryCurator(
        memory, scripted("Trabalho na Globant como dev sênior em Lisboa com Ana.")
    )
    assert await invents.normalize(note) == [note]


async def test_normalize_without_model_or_on_error_keeps_text(memory: VectorMemory) -> None:
    assert await MemoryCurator(memory, None).normalize("  moro em sp  ") == ["moro em sp"]

    class Broken(FakeListChatModel):
        async def ainvoke(self, *args: object, **kwargs: object):
            raise ConnectionError("ollama down")

    assert await MemoryCurator(memory, Broken(responses=["x"])).normalize("moro em sp") == [
        "moro em sp"
    ]


async def test_edit_splits_note_updating_in_place(memory: VectorMemory) -> None:
    (memory_id,) = await memory.add(["cachorro thor"], {"source": "manual"})
    curator = MemoryCurator(
        memory, scripted("Meu cachorro se chama Thor.\nMinha gata se chama Luna.")
    )

    result = await curator.edit(memory_id, "meu cachorro thor e minha gata luna")

    assert result.normalized is True
    assert [r.content for r in result.saved] == [
        "Meu cachorro se chama Thor.",
        "Minha gata se chama Luna.",
    ]
    assert result.saved[0].id == memory_id  # the first fact rewrites the edited memory
    assert result.saved[1].metadata["source"] == "manual"  # extra facts keep the source
    assert memory.count() == 2


async def test_edit_into_an_existing_fact_merges_into_it(memory: VectorMemory) -> None:
    existing, edited = await memory.add(["Moro em Recife.", "moro em recife"])
    curator = MemoryCurator(memory, scripted("Moro em Recife."))

    result = await curator.edit(edited, "moro em recife")

    assert [r.id for r in result.duplicates] == [existing]
    assert [r.id for r in result.archived] == [edited]
    assert memory.get(edited).metadata["superseded_by"] == existing


def test_api_optimize_flag(client: TestClient, memory: VectorMemory) -> None:
    llm = scripted("Meu cachorro se chama Thor.\nMinha gata se chama Luna.")
    app.dependency_overrides[get_curator] = lambda: MemoryCurator(memory, llm)

    created = client.post(
        "/memory", json={"texts": ["meu cachorro thor e minha gata luna"], "optimize": True}
    )
    assert created.status_code == 201
    body = created.json()
    assert body["normalized"] is True
    assert [r["content"] for r in body["saved"]] == [
        "Meu cachorro se chama Thor.",
        "Minha gata se chama Luna.",
    ]
    assert len(body["ids"]) == 2

    # Without optimize, the edit is stored exactly as written.
    (first,) = body["ids"][:1]
    raw = client.patch(f"/memory/{first}", json={"content": "cachorro: thor (vira-lata)"})
    assert raw.json()["memory"]["content"] == "cachorro: thor (vira-lata)"
    assert raw.json()["normalized"] is False
