from fastapi.testclient import TestClient

from origin.api.routes import setup
from origin.core.ollama import OllamaStatus


def events(response) -> list[tuple[str, dict]]:
    import json

    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def status(reachable=True, **models) -> OllamaStatus:
    return OllamaStatus(url="http://ollama:11434", reachable=reachable, models=models)


def test_missing_models_are_pulled_with_progress(client: TestClient, monkeypatch) -> None:
    async def fake_status(settings):
        return status(**{"qwen3:8b": True, "bge-m3": False})

    pulled = []

    async def fake_pull(settings, model):
        pulled.append(model)
        yield {"status": "pulling manifest"}
        yield {"status": "downloading", "total": 200, "completed": 50}
        yield {"status": "success"}

    monkeypatch.setattr(setup, "check_ollama", fake_status)
    monkeypatch.setattr(setup, "pull", fake_pull)

    got = events(client.post("/setup/pull"))

    assert pulled == ["bge-m3"]  # only what is missing
    assert (
        "progress",
        {"model": "bge-m3", "status": "downloading", "completed": 50, "total": 200, "percent": 25},
    ) in got
    assert got[-2:] == [("model_done", {"model": "bge-m3"}), ("done", {"models": ["bge-m3"]})]


def test_pull_errors_are_reported_in_band(client: TestClient, monkeypatch) -> None:
    async def fake_status(settings):
        return status(**{"qwen3:8b": False})

    async def fake_pull(settings, model):
        yield {"error": "pull model manifest: file does not exist"}

    monkeypatch.setattr(setup, "check_ollama", fake_status)
    monkeypatch.setattr(setup, "pull", fake_pull)
    got = events(client.post("/setup/pull"))
    assert got == [
        ("error", {"model": "qwen3:8b", "detail": "pull model manifest: file does not exist"})
    ]


def test_ollama_down(client: TestClient, monkeypatch) -> None:
    async def fake_status(settings):
        return status(reachable=False)

    monkeypatch.setattr(setup, "check_ollama", fake_status)
    assert events(client.post("/setup/pull"))[0][0] == "error"
    assert client.get("/setup/status").json()["reachable"] is False
