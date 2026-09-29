"""Which tools a turn offers, and how big that makes its fixed prompt."""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from origin.core import LLMEngine
from tests.fakes import ScriptedChatModel

has_documents = False


@tool
async def search_documents(query: str) -> str:
    """Search the company's documents, with a long description so the schema weighs more
    than the others: policies, manuals, contracts, spreadsheets, presentations."""
    return "nada"


@tool
async def web_search(query: str) -> str:
    """Search the web."""
    return "nada"


@tool
async def days_until(date: str) -> str:
    """Days until a date."""
    return "3"


@tool
async def broken(x: str) -> str:
    """A tool whose availability check fails."""
    return x


async def documents_available() -> bool:
    return has_documents


def check_fails() -> bool:
    raise RuntimeError("store down")


search_documents.metadata = {"available": documents_available}
web_search.metadata = {"network": True}
broken.metadata = {"available": check_fails}
TOOLS = {t.name: t for t in (search_documents, web_search, days_until, broken)}


def engine(model: ScriptedChatModel | None = None) -> LLMEngine:
    return LLMEngine(
        model or ScriptedChatModel(responses=[AIMessage("ok")]), "sys", "m", tools=TOOLS
    )


async def test_tools_are_offered_only_when_they_can_help() -> None:
    global has_documents
    has_documents = False
    subject = engine()
    # No documents yet: the knowledge tool is left out; a failing check keeps its tool.
    assert set(await subject.offered_tools()) == {"days_until", "broken"}
    assert set(await subject.offered_tools(use_web=True)) == {"days_until", "broken", "web_search"}
    has_documents = True
    assert "search_documents" in await subject.offered_tools()


async def test_the_model_is_not_given_unavailable_tools() -> None:
    global has_documents
    has_documents = False
    model = ScriptedChatModel(responses=[AIMessage("Olá!")])
    await engine(model).generate("Oi", use_memory=False)
    assert model.bound_tools == ["days_until", "broken"]


def test_the_fixed_prompt_counts_only_the_offered_tools() -> None:
    subject = engine()
    everything = subject.base_tokens(TOOLS)
    without = subject.base_tokens(["days_until", "broken", "web_search"])
    bare = subject.base_tokens([])
    assert bare < without < everything
    # The longest schema weighs the most.
    assert everything - without > subject.base_tokens(["days_until"]) - bare


async def test_the_measurement_is_split_among_the_tools() -> None:
    class Counting(ScriptedChatModel):
        def _generate(self, messages, *args, **kwargs):
            tokens = 1000 if self.bound_tools else 400
            self.responses = [AIMessage("ok", usage_metadata={
                "input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1
            })]  # fmt: skip
            return super()._generate(messages, *args, **kwargs)

    global has_documents
    has_documents = True
    subject = engine(Counting(responses=[AIMessage("ok")]))
    measured = await subject.measure_base_tokens()
    bare = subject.base_tokens([])
    assert 390 <= bare < 400  # the model's count, minus the probe message
    assert abs(subject.base_tokens(TOOLS) - bare - 600) <= 2  # rounding of the split
    assert measured == subject.base_tokens(["search_documents", "days_until", "broken"])
    # The workspaces' copies share the measurement.
    assert subject.with_memory(None, TOOLS).base_tokens([]) == bare
