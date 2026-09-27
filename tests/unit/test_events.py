import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from main import app
from origin.api.routes.chat import get_engine
from origin.core import LLMEngine
from tests.fakes import ScriptedChatModel, tool_call


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_events_stream_tool_calls_tokens_and_done(client: TestClient) -> None:
    model = ScriptedChatModel(responses=[tool_call("echo", {"text": "hi"}), AIMessage("feito")])
    engine = LLMEngine(model, "sys", "scripted", tools={"echo": echo})
    app.dependency_overrides[get_engine] = lambda: engine
    session_id = client.post("/sessions").json()["id"]

    response = client.post("/chat/events", json={"message": "oi", "session_id": session_id})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["tool_call", "token", "done"]
    assert events[0][1] == {"name": "echo", "args": {"text": "hi"}, "output": "echo: hi"}
    assert events[1][1] == {"text": "feito"}
    done = events[2][1]
    assert done["response"] == "feito"
    assert done["session_id"] == session_id
    assert [c["name"] for c in done["tool_calls"]] == ["echo"]

    stored = client.get(f"/sessions/{session_id}").json()["messages"]
    assert stored[1]["content"] == "feito"
    assert stored[1]["tool_calls"][0]["name"] == "echo"


class MidStreamFailure(LLMEngine):
    def __init__(self) -> None:
        self.model_name = "failing"

    async def events(self, *args: object, **kwargs: object):
        yield "parcial"
        raise RuntimeError("model crashed")


def test_events_report_mid_stream_errors_in_band(client: TestClient) -> None:
    app.dependency_overrides[get_engine] = MidStreamFailure

    response = client.post("/chat/events", json={"message": "oi"})

    assert response.status_code == 200
    assert parse_sse(response.text) == [
        ("token", {"text": "parcial"}),
        ("error", {"detail": "RuntimeError: model crashed"}),
    ]
