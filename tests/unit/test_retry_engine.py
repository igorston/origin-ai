from collections.abc import AsyncIterator
from typing import Any

import httpx
import ollama
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from origin.core import LLMEngine
from origin.memory import VectorMemory
from origin.memory.curator import MemoryCurator
from origin.retry import RetryPolicy, call_with_retry, is_mid_stream_failure

FAST = RetryPolicy(attempts=3, backoff=0)
# What Ollama reported when its runner died mid-reply (HTTP 200 already sent).
RUNNER_DIED = ollama.ResponseError(
    "error reading llama-server response: read tcp 127.0.0.1:58505->127.0.0.1:49966: wsarecv"
)


class FlakyChatModel(BaseChatModel):
    """Each call consumes one step: an exception to raise before any output, a
    (text, exception) pair to raise after streaming some text, or a reply text."""

    steps: list[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def _next(self) -> Any:
        self.calls += 1
        return self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        step = self._next()
        if isinstance(step, Exception):
            raise step
        return ChatResult(generations=[ChatGeneration(message=AIMessage(step))])

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        step = self._next()
        if isinstance(step, Exception):
            raise step
        text, error = step if isinstance(step, tuple) else (step, None)
        for word in text.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
        if error:
            raise error

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FlakyChatModel":
        return self


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (RUNNER_DIED, True),
        (httpx.ReadError("connection closed"), True),
        (httpx.RemoteProtocolError("peer closed"), True),
        # Start-of-request 5xx: already retried by the transport; don't multiply attempts.
        (ollama.ResponseError("health resp: dial tcp ...: connectex", 500), False),
        (ollama.ResponseError("model 'x' not found", 404), False),
        (ollama.ResponseError("invalid tool call format"), False),
        (ValueError("bug"), False),
    ],
)
def test_is_mid_stream_failure(exc: Exception, retryable: bool) -> None:
    assert is_mid_stream_failure(exc) is retryable


async def test_call_with_retry() -> None:
    attempts = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RUNNER_DIED
        return "ok"

    assert await call_with_retry(flaky, FAST) == "ok"
    assert len(attempts) == 3

    async def broken() -> str:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        await call_with_retry(broken, FAST)


def engine(model: FlakyChatModel, **kwargs: Any) -> LLMEngine:
    return LLMEngine(model, "sys", "flaky", retry=FAST, **kwargs)


async def test_reply_retried_when_it_fails_before_any_text() -> None:
    model = FlakyChatModel(steps=[RUNNER_DIED, "Olá, tudo bem?"])

    chunks = [c async for c in engine(model).stream("oi", use_memory=False)]

    assert "".join(chunks).strip() == "Olá, tudo bem?"
    assert model.calls == 2


async def test_reply_not_retried_once_text_was_shown() -> None:
    model = FlakyChatModel(steps=[("Olá, tudo", RUNNER_DIED), "nunca"])
    shown = []

    with pytest.raises(ollama.ResponseError):
        async for chunk in engine(model).stream("oi", use_memory=False):
            shown.append(chunk)

    assert "".join(shown).strip() == "Olá, tudo"  # no second reply spliced in
    assert model.calls == 1


async def test_reply_gives_up_after_the_policy_attempts() -> None:
    model = FlakyChatModel(steps=[RUNNER_DIED])
    with pytest.raises(ollama.ResponseError):
        await engine(model).generate("oi", use_memory=False)
    assert model.calls == 3


async def test_routing_turn_is_retried() -> None:
    router = FlakyChatModel(steps=[RUNNER_DIED, "NONE"])
    model = FlakyChatModel(steps=["resposta"])
    from langchain_core.tools import tool

    @tool
    def noop() -> str:
        """Does nothing."""
        return ""

    result = await engine(model, router=router, tools={"noop": noop}, tool_routing=True).generate(
        "oi", use_memory=False
    )

    assert result.text.strip() == "resposta"
    assert router.calls == 2


async def test_curator_model_calls_are_retried(memory: VectorMemory) -> None:
    llm = FlakyChatModel(steps=[RUNNER_DIED, "Meu cachorro se chama Thor."])
    curator = MemoryCurator(memory, llm, retry=FAST)

    assert await curator.normalize("meu cachorro thor") == ["Meu cachorro se chama Thor."]
    assert llm.calls == 2
