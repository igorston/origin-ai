from fastapi.testclient import TestClient

from origin.core import ChatTurn, LLMEngine
from origin.memory import VectorMemory


def test_chat(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Oi"})
    assert response.status_code == 200
    assert response.json() == {
        "response": "Olá, eu sou o Origin.",
        "model": "fake",
        "session_id": None,
        "tool_calls": [],
    }


def test_chat_stream(client: TestClient) -> None:
    response = client.post("/chat/stream", json={"message": "Oi"})
    assert response.status_code == 200
    assert response.text == "Olá, eu sou o Origin."


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


async def test_build_messages_includes_system_and_history(fake_engine: LLMEngine) -> None:
    history = [ChatTurn(role="user", content="a"), ChatTurn(role="assistant", content="b")]
    messages = await fake_engine._build_messages("c", history)
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert [m.content for m in messages] == ["test", "a", "b", "c"]


async def test_build_messages_injects_recalled_memory(
    fake_engine: LLMEngine, memory: VectorMemory
) -> None:
    # Fake embeddings are hash-based, so only an identical query is guaranteed to match.
    fact = "Meu cachorro se chama Thor."
    await memory.add([fact])

    messages = await fake_engine._build_messages(fact)
    assert "Long-term memory" in messages[0].content
    assert f"- {fact}" in messages[0].content

    without = await fake_engine._build_messages(fact, use_memory=False)
    assert without[0].content == "test"


async def test_recall_failure_falls_back_to_plain_prompt(fake_engine: LLMEngine) -> None:
    class BrokenMemory:
        async def search(self, *args: object, **kwargs: object) -> list:
            raise ConnectionError("ollama offline")

    fake_engine.memory = BrokenMemory()  # type: ignore[assignment]
    messages = await fake_engine._build_messages("oi")
    assert messages[0].content == "test"
