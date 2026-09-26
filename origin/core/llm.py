import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from itertools import chain
from typing import Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from origin.config import Settings
from origin.core.agent import ToolCallRecord, execute_tool_calls, unbacked_claims
from origin.memory import MemoryHit, VectorMemory
from origin.prompts import load_prompt

logger = logging.getLogger(__name__)

# How much of the previous exchange is prepended to the memory query for follow-ups.
RECALL_CONTEXT_CHARS = 500


def make_chat_model(settings: Settings, temperature: float | None = None) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature if temperature is None else temperature,
        reasoning=settings.ollama_reasoning,
        keep_alive=settings.ollama_keep_alive,
    )


def remind_language(tool_messages: list[ToolMessage], message: str) -> list[ToolMessage]:
    """Tool results are English and are the last thing the model reads before replying, which
    made small models answer in English. Quoting the user's own words anchors the language."""
    if tool_messages:
        quote = message if len(message) <= 120 else f"{message[:117]}..."
        tool_messages[
            -1
        ].content += f'\n\n[Reply to the user in the same language as their message: "{quote}"]'
    return tool_messages


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatResult(BaseModel):
    text: str
    tool_calls: list[ToolCallRecord] = []


class LLMEngine:
    """Model-agnostic chat engine with optional RAG memory and a tool-calling loop."""

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        model_name: str,
        memory: VectorMemory | None = None,
        memory_top_k: int = 4,
        memory_min_score: float = 0.45,
        tools: Mapping[str, BaseTool] | None = None,
        max_tool_iterations: int = 5,
        tool_routing: bool = False,
        contextual_recall: bool = True,
        router: BaseChatModel | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.memory = memory
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score
        self.tools = dict(tools or {})
        self.max_tool_iterations = max_tool_iterations
        self.tool_routing = tool_routing
        self.contextual_recall = contextual_recall
        # Model for the routing turn; a deterministic copy makes tool decisions consistent.
        self.router = router or model

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        memory: VectorMemory | None = None,
        tools: Mapping[str, BaseTool] | None = None,
    ) -> "LLMEngine":
        return cls(
            make_chat_model(settings),
            load_prompt("system"),
            settings.ollama_model,
            memory=memory,
            memory_top_k=settings.memory_top_k,
            memory_min_score=settings.memory_min_score,
            tools=tools,
            max_tool_iterations=settings.agent_max_tool_iterations,
            tool_routing=settings.agent_tool_routing,
            contextual_recall=settings.memory_contextual_recall,
            router=make_chat_model(settings, temperature=settings.agent_routing_temperature),
        )

    def _bind_tools(
        self, use_tools: bool
    ) -> tuple[Runnable[LanguageModelInput, BaseMessage], dict[str, BaseTool]]:
        if not (use_tools and self.tools):
            return self.model, {}
        try:
            return self.model.bind_tools(list(self.tools.values())), self.tools
        except NotImplementedError:
            logger.warning("Model %s does not support tool calling", self.model_name)
            return self.model, {}

    async def _recall(self, message: str, history: Sequence[ChatTurn] = ()) -> str | None:
        if self.memory is None:
            return None
        # Follow-ups like "and her birthday?" carry no entity on their own, so also search with
        # the last exchange prepended, and keep each memory's best score across both queries.
        queries = [message]
        if history and self.contextual_recall:
            recent = "\n".join(turn.content for turn in history[-2:])
            queries.append(f"{recent[-RECALL_CONTEXT_CHARS:]}\n{message}")
        try:
            results = await asyncio.gather(
                *(
                    self.memory.search(query, k=self.memory_top_k, min_score=self.memory_min_score)
                    for query in queries
                )
            )
        except Exception:
            logger.warning("Memory recall failed; answering without memory", exc_info=True)
            return None
        best: dict[str, MemoryHit] = {}
        for hit in chain.from_iterable(results):
            if hit.id not in best or hit.score > best[hit.id].score:
                best[hit.id] = hit
        hits = sorted(best.values(), key=lambda hit: hit.score, reverse=True)[: self.memory_top_k]
        if not hits:
            return None
        memories = "\n".join(f"- {hit.content}" for hit in hits)
        return load_prompt("memory_context").format(memories=memories)

    async def _build_messages(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        with_tools: bool = False,
    ) -> list[BaseMessage]:
        sections = [self.system_prompt]
        if with_tools:
            sections.append(load_prompt("tools"))
        if use_memory and (context := await self._recall(message, history)):
            sections.append(context)

        messages: list[BaseMessage] = [SystemMessage("\n\n".join(sections))]
        for turn in history:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.content))
        messages.append(HumanMessage(message))
        return messages

    async def events(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
    ) -> AsyncIterator[str | ToolCallRecord]:
        """Agent loop: yields text chunks as they stream and a record per executed tool call."""
        bound, tools = self._bind_tools(use_tools)
        messages = await self._build_messages(message, history, use_memory, bool(tools))
        executed: list[ToolCallRecord] = []

        async def run_tools(tool_calls: list[ToolCall]) -> list[ToolCallRecord]:
            tool_messages, records = await execute_tool_calls(tool_calls, tools)
            messages.extend(remind_language(tool_messages, message))
            executed.extend(records)
            return records

        first_iteration = 0
        if tools and self.tool_routing:
            # Small models stop calling tools once they start writing text, so for compound
            # messages they answer the easy part and drop the rest. A dedicated routing turn,
            # where answering is not allowed, makes them commit to every needed tool first.
            system = f"{messages[0].content}\n\n{load_prompt('tool_routing')}"
            router = self.router.bind_tools(list(tools.values()))
            decision = await router.ainvoke([SystemMessage(system), *messages[1:]])
            if decision.tool_calls:
                messages.append(AIMessage("", tool_calls=decision.tool_calls))
                for record in await run_tools(decision.tool_calls):
                    yield record
                first_iteration = 1

        for iteration in range(first_iteration, self.max_tool_iterations + 1):
            # On the last iteration drop the tools so the model is forced to answer.
            final = iteration == self.max_tool_iterations
            model = self.model if final else bound
            response = None
            async for chunk in model.astream(messages):
                response = chunk if response is None else response + chunk
                if chunk.text:
                    yield chunk.text

            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls and tools and not final:
                messages.append(response)
                for record in await run_tools(tool_calls):
                    yield record
                continue

            if tool_calls:
                logger.warning("Tool iteration limit reached; ignoring %d call(s)", len(tool_calls))
            if response is not None and tools:
                for record in await self._verify_claims(tools, messages, response, executed):
                    yield record
            return

    async def _verify_claims(
        self,
        tools: Mapping[str, BaseTool],
        messages: list[BaseMessage],
        response: BaseMessage,
        executed: list[ToolCallRecord],
    ) -> list[ToolCallRecord]:
        """If the final reply claims an effect ("anotei!") whose tool was never called, give
        the model one chance to actually call it, so the claim becomes true."""
        claimed = unbacked_claims(response.text, executed, set(tools))
        if not claimed:
            return []
        logger.warning("Reply claims %s without calling it; verifying", sorted(claimed))
        check = HumanMessage(load_prompt("claim_check").format(tools=", ".join(sorted(claimed))))
        router = self.router.bind_tools(list(tools.values()))
        decision = await router.ainvoke([*messages, AIMessage(response.text), check])
        calls = [call for call in decision.tool_calls if call["name"] in claimed]
        if not calls:
            return []
        _, records = await execute_tool_calls(calls, tools)
        return records

    async def generate(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
    ) -> ChatResult:
        text: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        async for event in self.events(message, history, use_memory, use_tools):
            if isinstance(event, str):
                text.append(event)
            else:
                tool_calls.append(event)
        return ChatResult(text="".join(text), tool_calls=tool_calls)

    async def stream(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
    ) -> AsyncIterator[str]:
        async for event in self.events(message, history, use_memory, use_tools):
            if isinstance(event, str):
                yield event
