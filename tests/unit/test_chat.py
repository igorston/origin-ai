from fastapi.testclient import TestClient

from origin.core import ChatTurn, LLMEngine


def test_chat(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Oi"})
    assert response.status_code == 200
    assert response.json() == {"response": "Olá, eu sou o Origin.", "model": "fake"}


def test_chat_stream(client: TestClient) -> None:
    response = client.post("/chat/stream", json={"message": "Oi"})
    assert response.status_code == 200
    assert response.text == "Olá, eu sou o Origin."


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_build_messages_includes_system_and_history(fake_engine: LLMEngine) -> None:
    history = [ChatTurn(role="user", content="a"), ChatTurn(role="assistant", content="b")]
    messages = fake_engine._build_messages("c", history)
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert [m.content for m in messages] == ["test", "a", "b", "c"]
