"""Automatic retries for transient Ollama failures, at the HTTP transport layer.

Observed in practice: Ollama answers 500 when its model-runner subprocess is briefly
unreachable ('health resp: Get "http://127.0.0.1:PORT/health": dial tcp ... connectex',
'Post ".../tokenize": dial tcp ...') and recovers by itself a moment later. These
failures happen before any token is streamed, so re-sending the request is safe.

Retrying in the transport covers every client in one place (chat, router, judge,
embeddings; sync and async) without touching LangChain.
"""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import httpx

if TYPE_CHECKING:
    from origin.config import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Bodies of 5xx responses that mean "try again", not "this request is wrong".
TRANSIENT_ERROR = re.compile(
    r"connectex|dial tcp|health resp|connection refused|connection reset|"
    r"runner process (has )?(terminated|exited)|server busy|EOF|"
    # Runner killed mid-reply: Ollama already answered 200 and reports it in the stream.
    r"error reading llama-server response|wsarecv|forcibly closed",
    re.IGNORECASE,
)
RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)
# Failures that surface after the response started, which the transport cannot retry.
MID_STREAM_EXCEPTIONS = (httpx.RemoteProtocolError, httpx.ReadError)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3  # total tries, including the first
    backoff: float = 0.5  # seconds; doubles after each failure

    def delay(self, failures: int) -> float:
        return self.backoff * 2 ** (failures - 1)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RetryPolicy":
        return cls(
            attempts=max(1, settings.ollama_retry_attempts),
            backoff=settings.ollama_retry_backoff,
        )


DEFAULT_POLICY = RetryPolicy()


def _is_transient(status: int, body: bytes) -> bool:
    if status == 503:
        return True
    return status in (500, 502) and bool(TRANSIENT_ERROR.search(body.decode(errors="replace")))


def _rebuild(response: httpx.Response, body: bytes, request: httpx.Request) -> httpx.Response:
    # The body was consumed to inspect it; hand the caller an equivalent fresh response.
    return httpx.Response(
        response.status_code, headers=response.headers, content=body, request=request
    )


def _log_retry(request: httpx.Request, failures: int, policy: RetryPolicy, reason: str) -> None:
    logger.warning(
        "Transient Ollama failure on %s (%s); retry %d/%d in %.1fs",
        request.url.path,
        reason[:160],
        failures,
        policy.attempts - 1,
        policy.delay(failures),
    )


class RetryingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(
        self, inner: httpx.AsyncBaseTransport | None = None, policy: RetryPolicy = DEFAULT_POLICY
    ) -> None:
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.policy = policy

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(1, self.policy.attempts + 1):
            last = attempt == self.policy.attempts
            try:
                response = await self.inner.handle_async_request(request)
            except RETRYABLE_EXCEPTIONS as exc:
                if last:
                    raise
                _log_retry(request, attempt, self.policy, repr(exc))
                await asyncio.sleep(self.policy.delay(attempt))
                continue
            if response.status_code < 500:
                return response
            body = await response.aread()
            await response.aclose()
            if last or not _is_transient(response.status_code, body):
                return _rebuild(response, body, request)
            _log_retry(request, attempt, self.policy, body.decode(errors="replace"))
            await asyncio.sleep(self.policy.delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        await self.inner.aclose()


class RetryingTransport(httpx.BaseTransport):
    """Sync twin, for the embedding client Chroma calls from a worker thread."""

    def __init__(
        self, inner: httpx.BaseTransport | None = None, policy: RetryPolicy = DEFAULT_POLICY
    ) -> None:
        self.inner = inner or httpx.HTTPTransport()
        self.policy = policy

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(1, self.policy.attempts + 1):
            last = attempt == self.policy.attempts
            try:
                response = self.inner.handle_request(request)
            except RETRYABLE_EXCEPTIONS as exc:
                if last:
                    raise
                _log_retry(request, attempt, self.policy, repr(exc))
                time.sleep(self.policy.delay(attempt))
                continue
            if response.status_code < 500:
                return response
            body = response.read()
            response.close()
            if last or not _is_transient(response.status_code, body):
                return _rebuild(response, body, request)
            _log_retry(request, attempt, self.policy, body.decode(errors="replace"))
            time.sleep(self.policy.delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        self.inner.close()


def is_mid_stream_failure(exc: BaseException) -> bool:
    """A transient failure raised while reading a response that had already started (the
    runner died mid-reply). Start-of-request 5xx errors are excluded: the transport has
    already retried those, and retrying again here would multiply the attempts."""
    if isinstance(exc, MID_STREAM_EXCEPTIONS):
        return True
    # ollama.ResponseError from an error chunk inside a 200 stream has status_code -1.
    status = getattr(exc, "status_code", None)
    return status == -1 and bool(TRANSIENT_ERROR.search(str(exc)))


async def call_with_retry(
    call: Callable[[], Awaitable[T]], policy: RetryPolicy, what: str = "model call"
) -> T:
    """Re-run a whole model call (routing, judge, normalization...) whose result is not
    shown to the user until it completes, so repeating it is invisible and safe."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt == policy.attempts or not is_mid_stream_failure(exc):
                raise
            logger.warning(
                "%s failed mid-stream (%s); retry %d/%d in %.1fs",
                what, str(exc)[:160], attempt, policy.attempts - 1, policy.delay(attempt),
            )  # fmt: skip
            await asyncio.sleep(policy.delay(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


def ollama_client_kwargs(policy: RetryPolicy) -> dict[str, dict]:
    """Keyword arguments for ChatOllama / OllamaEmbeddings that enable retries."""
    return {
        "sync_client_kwargs": {"transport": RetryingTransport(policy=policy)},
        "async_client_kwargs": {"transport": RetryingAsyncTransport(policy=policy)},
    }
