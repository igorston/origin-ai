from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from origin.core import LLMEngine
from tests.fakes import ScriptedChatModel, tool_call


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@tool
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


def make_engine(*responses: AIMessage, max_iterations: int = 5) -> LLMEngine:
    model = ScriptedChatModel(responses=list(responses))
    return LLMEngine(
        model,
        system_prompt="sys",
        model_name="scripted",
        tools={t.name: t for t in (echo, explode)},
        max_tool_iterations=max_iterations,
    )


async def test_tool_loop_executes_call_and_returns_final_answer() -> None:
    engine = make_engine(tool_call("echo", {"text": "hi"}), AIMessage("done"))

    result = await engine.generate("say hi", use_memory=False)

    assert result.text == "done"
    assert [(c.name, c.args, c.output) for c in result.tool_calls] == [
        ("echo", {"text": "hi"}, "echo: hi")
    ]
    second_call = engine.model.received[1]
    assert isinstance(second_call[-1], ToolMessage)
    assert second_call[-1].content == "echo: hi"
    assert engine.model.bound_tools == ["echo", "explode"]
    assert "## Tools" in second_call[0].content


async def test_tool_errors_are_reported_to_model() -> None:
    engine = make_engine(
        tool_call("explode", call_id="a"), tool_call("missing", call_id="b"), AIMessage("ok")
    )

    result = await engine.generate("go", use_memory=False)

    assert result.text == "ok"
    assert result.tool_calls[0].output == "Error: RuntimeError: boom"
    assert result.tool_calls[1].output.startswith("Error: unknown tool 'missing'")


async def test_iteration_limit_stops_the_loop() -> None:
    engine = make_engine(tool_call("echo", {"text": "again"}), max_iterations=2)

    result = await engine.generate("loop", use_memory=False)

    assert len(result.tool_calls) == 2
    assert len(engine.model.received) == 3


async def test_use_tools_false_skips_binding_and_tool_prompt() -> None:
    engine = make_engine(AIMessage("plain"))

    result = await engine.generate("hi", use_memory=False, use_tools=False)

    assert result.text == "plain"
    assert engine.model.bound_tools == []
    assert engine.model.received[0][0].content == "sys"


async def test_stream_yields_only_text() -> None:
    engine = make_engine(tool_call("echo", {"text": "x"}), AIMessage("final"))

    chunks = [chunk async for chunk in engine.stream("hi", use_memory=False)]

    assert "".join(chunks) == "final"
