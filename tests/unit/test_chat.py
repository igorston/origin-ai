from fastapi.testclient import TestClient

from origin.core import ChatTurn, LLMEngine
from origin.memory import VectorMemory


def test_chat(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Oi"})
    assert response.status_code == 200
    body = response.json()
    assert {k: body[k] for k in ("response", "model", "session_id", "tool_calls")} == {
        "response": "Olá, eu sou o Origin.",
        "model": "fake",
        "session_id": None,
        "tool_calls": [],
    }
    assert body["context"]["state"] == "ok"
    assert body["compactions"] == [] and body["trimmed"] == 0


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
    # Recalled memories ride on the user message; the system prompt stays static (cacheable).
    assert messages[0].content == "test"
    assert "Long-term memory" in messages[-1].content
    assert f"- {fact}" in messages[-1].content
    assert messages[-1].content.endswith(f"[User message]\n\n{fact}")

    without = await fake_engine._build_messages(fact, use_memory=False)
    assert without[-1].content == fact


async def test_recall_failure_falls_back_to_plain_prompt(fake_engine: LLMEngine) -> None:
    class BrokenMemory:
        async def search(self, *args: object, **kwargs: object) -> list:
            raise ConnectionError("ollama offline")

    fake_engine.memory = BrokenMemory()  # type: ignore[assignment]
    messages = await fake_engine._build_messages("oi")
    assert messages[0].content == "test"


class MeasuredEngine:
    """An engine whose answer came with Ollama's token counts, after tools or not."""

    model_name = "fake"

    def __init__(self, tool_calls: list) -> None:
        self.tool_calls = tool_calls

    async def offered_tools(self, use_web: bool = False) -> dict:
        return {}

    def base_tokens(self, tools) -> int:
        return 50

    async def generate(self, message, history=(), *args, **kwargs):
        from origin.core.agent import ToolCallRecord
        from origin.core.llm import ChatResult, TurnUsage

        calls = [ToolCallRecord(name=n, args={}, output="página " * 800) for n in self.tool_calls]
        return ChatResult(
            text="Resposta.", tool_calls=calls, usage=TurnUsage(input_tokens=4000, output_tokens=20)
        )


async def test_a_turn_with_tools_is_estimated_not_measured(client: TestClient, sessions) -> None:
    from main import app
    from origin.api.routes.chat import get_engine

    # The measurement of a web turn counts the fetched page, which the next turn drops.
    app.dependency_overrides[get_engine] = lambda: MeasuredEngine(["fetch_url"])
    session_id = client.post("/sessions").json()["id"]
    after = client.post("/chat", json={"message": "Explique", "session_id": session_id}).json()

    assert not after["context"]["measured"] and after["context"]["used"] < 4000
    assert (await sessions.get(session_id)).context_tokens == 0  # nothing to calibrate on

    # Without tools the measurement is the conversation itself: kept.
    app.dependency_overrides[get_engine] = lambda: MeasuredEngine([])
    after = client.post("/chat", json={"message": "E agora?", "session_id": session_id}).json()
    assert after["context"]["measured"] and after["context"]["used"] == 4020
    assert (await sessions.get(session_id)).context_tokens == 4020
