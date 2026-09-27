import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from main import app
from origin.api.routes.chat import get_engine
from origin.core import ChatTurn, LLMEngine
from origin.core.llm import split_clauses
from origin.memory import MemoryHit
from origin.memory.storage import SessionStore
from tests.fakes import ScriptedChatModel


async def test_store_roundtrip(sessions: SessionStore) -> None:
    session = await sessions.create()
    await sessions.append(session.id, "user", "Primeira pergunta " + "x" * 100)
    await sessions.append(session.id, "assistant", "resposta", [{"name": "t", "args": {}}])
    await sessions.append(session.id, "user", "segunda")

    stored = await sessions.get(session.id)
    assert stored.title == ("Primeira pergunta " + "x" * 100)[:60]
    assert stored.message_count == 3

    messages = await sessions.messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[1].tool_calls == [{"name": "t", "args": {}}]

    last_two = await sessions.messages(session.id, limit=2)
    assert [m.content for m in last_two] == ["resposta", "segunda"]


async def test_store_list_and_delete(sessions: SessionStore) -> None:
    older = await sessions.create("older")
    newer = await sessions.create("newer")
    await sessions.append(older.id, "user", "bump")  # most recently updated first

    assert [s.id for s in await sessions.list()] == [older.id, newer.id]
    assert await sessions.delete(older.id) is True
    assert await sessions.delete(older.id) is False
    assert await sessions.messages(older.id) == []  # cascade


def test_session_api_lifecycle(client: TestClient) -> None:
    created = client.post("/sessions", json={"title": "Planejamento"})
    assert created.status_code == 201
    session_id = created.json()["id"]

    chat = client.post("/chat", json={"message": "Oi", "session_id": session_id})
    assert chat.status_code == 200
    assert chat.json()["session_id"] == session_id

    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["title"] == "Planejamento"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "Oi"),
        ("assistant", "Olá, eu sou o Origin."),
    ]
    assert [s["id"] for s in client.get("/sessions").json()] == [session_id]

    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.get(f"/sessions/{session_id}").status_code == 404
    assert client.delete(f"/sessions/{session_id}").status_code == 404


def test_session_history_is_sent_to_the_model(client: TestClient) -> None:
    model = ScriptedChatModel(responses=[AIMessage("primeira"), AIMessage("segunda")])
    app.dependency_overrides[get_engine] = lambda: LLMEngine(model, "sys", "scripted")
    session_id = client.post("/sessions").json()["id"]

    client.post("/chat", json={"message": "Minha esposa se chama Ana.", "session_id": session_id})
    client.post("/chat", json={"message": "Como ela se chama?", "session_id": session_id})

    second_call = model.received[1]
    assert [(m.type, m.content) for m in second_call[1:]] == [
        ("human", "Minha esposa se chama Ana."),
        ("ai", "primeira"),
        ("human", "Como ela se chama?"),
    ]


def test_stream_persists_exchange(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["id"]

    response = client.post("/chat/stream", json={"message": "Oi", "session_id": session_id})

    assert response.text == "Olá, eu sou o Origin."
    messages = client.get(f"/sessions/{session_id}").json()["messages"]
    assert [m["content"] for m in messages] == ["Oi", "Olá, eu sou o Origin."]


def test_chat_rejects_unknown_session_and_mixed_history(client: TestClient) -> None:
    assert client.post("/chat", json={"message": "Oi", "session_id": "nope"}).status_code == 404
    mixed = {"message": "Oi", "session_id": "x", "history": [{"role": "user", "content": "a"}]}
    assert client.post("/chat", json=mixed).status_code == 422


@pytest.mark.parametrize(
    ("message", "clauses"),
    [
        (
            "Onde eu moro e quantos dias faltam pro Natal?",
            ["Onde eu moro", "quantos dias faltam pro Natal"],
        ),
        (
            "Que horas são agora, e qual o nome do meu cachorro?",
            ["Que horas são agora", "qual o nome do meu cachorro"],
        ),
        ("Onde eu moro?", []),
        ("Tom e Jerry", []),  # single-word parts are not clauses
    ],
)
def test_split_clauses(message: str, clauses: list[str]) -> None:
    assert split_clauses(message) == clauses


class RecordingMemory:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, k: int, min_score: float) -> list[MemoryHit]:
        self.queries.append(query)
        # The contextual query finds the fact with a better score than the bare one.
        score = 0.8 if "Júlia" in query else 0.5
        return [MemoryHit(id="1", content="Júlia adora chocolate.", score=score, metadata={})]


@pytest.mark.parametrize("with_history", [False, True])
async def test_recall_uses_previous_exchange(with_history: bool) -> None:
    memory = RecordingMemory()
    engine = LLMEngine(ScriptedChatModel(responses=[AIMessage("")]), "sys", "m", memory=memory)
    history = (
        [
            ChatTurn(role="user", content="Minha irmã se chama Júlia."),
            ChatTurn(role="assistant", content="Que legal!"),
        ]
        if with_history
        else []
    )

    context = await engine._recall("Do que ela gosta?", history)

    assert "- Júlia adora chocolate." in context  # deduplicated across queries
    assert context.count("Júlia adora chocolate") == 1
    if with_history:
        assert memory.queries == [
            "Do que ela gosta?",
            "Minha irmã se chama Júlia.\nQue legal!\nDo que ela gosta?",
        ]
    else:
        assert memory.queries == ["Do que ela gosta?"]
