import httpx
import ollama
import pytest
from fastapi.testclient import TestClient

from origin import __version__
from origin.api.routes import health
from origin.core import LLMEngine
from origin.core.ollama import OllamaStatus


def fake_status(**kwargs: object):
    async def check(settings: object) -> OllamaStatus:
        return OllamaStatus(url="http://ollama", **kwargs)

    return check


@pytest.mark.parametrize("loaded", [True, False])
def test_health_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch, loaded: bool) -> None:
    status = fake_status(reachable=True, models={"m": True}, loaded={"m": loaded})
    monkeypatch.setattr(health, "check_ollama", status)

    response = client.get("/health")

    # Unloaded models make the next reply slow, not the service unhealthy.
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "ready": loaded,
        "ollama": {
            "url": "http://ollama",
            "reachable": True,
            "models": {"m": True},
            "loaded": {"m": loaded},
        },
    }


@pytest.mark.parametrize(
    "status", [{"reachable": False}, {"reachable": True, "models": {"m": False}}]
)
def test_health_degraded(client: TestClient, monkeypatch: pytest.MonkeyPatch, status: dict) -> None:
    monkeypatch.setattr(health, "check_ollama", fake_status(**status))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


class FailingEngine(LLMEngine):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.model_name = "failing"

    async def generate(self, *args: object, **kwargs: object):
        raise self.exc

    async def events(self, *args: object, **kwargs: object):
        raise self.exc
        yield  # pragma: no cover


@pytest.mark.parametrize(
    ("exc", "status", "fragment"),
    [
        (httpx.ConnectError("refused"), 503, "not reachable"),
        (ConnectionError("Failed to connect"), 503, "not reachable"),
        (ollama.ResponseError("model 'x' not found", 404), 502, "ollama pull"),
    ],
)
@pytest.mark.parametrize("path", ["/chat", "/chat/stream", "/chat/events"])
def test_backend_errors_become_http_errors(
    client: TestClient, path: str, exc: Exception, status: int, fragment: str
) -> None:
    from main import app
    from origin.api.routes.chat import get_engine

    app.dependency_overrides[get_engine] = lambda: FailingEngine(exc)

    response = client.post(path, json={"message": "oi"})

    assert response.status_code == status
    assert fragment in response.json()["detail"]
