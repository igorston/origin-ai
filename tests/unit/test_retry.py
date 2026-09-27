import json

import httpx
import pytest
from langchain_ollama import ChatOllama, OllamaEmbeddings

from origin.config import Settings
from origin.core import make_chat_model
from origin.memory import VectorMemory
from origin.retry import (
    RetryingAsyncTransport,
    RetryingTransport,
    RetryPolicy,
    ollama_client_kwargs,
)

# The exact failure Ollama returned several times during testing.
RUNNER_DOWN = (
    '{"error":"health resp: Get \\"http://127.0.0.1:63347/health\\": dial tcp '
    '127.0.0.1:63347: connectex: Uma tentativa de conexão falhou"}'
)
FAST = RetryPolicy(attempts=3, backoff=0)


class Script:
    """Mock Ollama: returns the scripted outcomes in order and counts calls."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return httpx.Response(status, content=body.encode(), request=request)


async def send_async(script: Script) -> httpx.Response:
    transport = RetryingAsyncTransport(httpx.MockTransport(script), FAST)
    async with httpx.AsyncClient(transport=transport, base_url="http://ollama") as client:
        return await client.post("/api/chat", json={"model": "m"})


def send_sync(script: Script) -> httpx.Response:
    transport = RetryingTransport(httpx.MockTransport(script), FAST)
    with httpx.Client(transport=transport, base_url="http://ollama") as client:
        return client.post("/api/embed", json={"model": "m"})


@pytest.fixture(params=["async", "sync"])
async def send(request: pytest.FixtureRequest):
    if request.param == "async":
        return send_async

    async def sync(script: Script) -> httpx.Response:
        return send_sync(script)

    return sync


async def test_retries_transient_500_then_succeeds(send) -> None:
    script = Script((500, RUNNER_DOWN), (200, '{"ok": true}'))
    response = await send(script)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert script.calls == 2


async def test_retries_503_and_dropped_connections(send) -> None:
    script = Script((503, "busy"), httpx.ConnectError("refused"), (200, "{}"))
    assert (await send(script)).status_code == 200
    assert script.calls == 3


async def test_does_not_retry_real_errors(send) -> None:
    script = Script((500, '{"error":"invalid tool schema"}'))
    response = await send(script)
    assert response.status_code == 500
    assert response.json() == {"error": "invalid tool schema"}  # body preserved
    assert script.calls == 1

    not_found = Script((404, '{"error":"model not found"}'))
    assert (await send(not_found)).status_code == 404
    assert not_found.calls == 1


async def test_gives_up_after_the_last_attempt(send) -> None:
    script = Script((500, RUNNER_DOWN))
    response = await send(script)
    assert response.status_code == 500
    assert "connectex" in response.text
    assert script.calls == 3

    down = Script(httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await send(down)
    assert down.calls == 3


def test_backoff_doubles() -> None:
    policy = RetryPolicy(attempts=4, backoff=0.5)
    assert [policy.delay(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]
    settings = Settings(ollama_retry_attempts=0, ollama_retry_backoff=0.1)
    assert RetryPolicy.from_settings(settings) == RetryPolicy(attempts=1, backoff=0.1)


def test_origin_clients_are_built_with_retries() -> None:
    chat = make_chat_model(Settings())
    assert isinstance(chat._async_client._client._transport, RetryingAsyncTransport)
    assert isinstance(chat._client._client._transport, RetryingTransport)
    embeddings = VectorMemory.from_settings(
        Settings(chroma_persist_dir=str(pytest.importorskip("tempfile").mkdtemp()))
    )._store.embeddings
    assert isinstance(embeddings._client._client._transport, RetryingTransport)


def ndjson(*objects: dict) -> str:
    return "\n".join(json.dumps(o) for o in objects) + "\n"


async def test_chat_model_recovers_from_runner_crash_end_to_end() -> None:
    reply = ndjson(
        {"model": "m", "created_at": "2026-01-01T00:00:00Z",
         "message": {"role": "assistant", "content": "Olá!"}, "done": False},
        {"model": "m", "created_at": "2026-01-01T00:00:00Z",
         "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    )  # fmt: skip
    script = Script((500, RUNNER_DOWN), (200, reply))
    transport = RetryingAsyncTransport(httpx.MockTransport(script), FAST)
    chat = ChatOllama(model="m", async_client_kwargs={"transport": transport})

    result = await chat.ainvoke("oi")

    assert result.text == "Olá!"
    assert script.calls == 2


def test_embeddings_recover_from_runner_crash_end_to_end() -> None:
    script = Script(
        (500, RUNNER_DOWN), (200, json.dumps({"model": "m", "embeddings": [[0.1, 0.2]]}))
    )
    kwargs = ollama_client_kwargs(FAST)
    kwargs["sync_client_kwargs"] = {
        "transport": RetryingTransport(httpx.MockTransport(script), FAST)
    }
    embeddings = OllamaEmbeddings(model="m", **kwargs)

    assert embeddings.embed_query("oi") == [0.1, 0.2]
    assert script.calls == 2
