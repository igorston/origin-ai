import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from origin.config import Settings
from origin.core.agent import ToolCallRecord, execute_tool_calls
from origin.memory import VectorMemory
from origin.prompts import load_prompt

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        memory: VectorMemory | None = None,
        tools: Mapping[str, BaseTool] | None = None,
    ) -> "LLMEngine":
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            reasoning=settings.ollama_reasoning,
            keep_alive=settings.ollama_keep_alive,
        )
        return cls(
            model,
            load_prompt("system"),
            settings.ollama_model,
            memory=memory,
            memory_top_k=settings.memory_top_k,
            memory_min_score=settings.memory_min_score,
            tools=tools,
            max_tool_iterations=settings.agent_max_tool_iterations,
            tool_routing=settings.agent_tool_routing,
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

    async def _recall(self, message: str) -> str | None:
        if self.memory is None:
            return None
        try:
            hits = await self.memory.search(
                message, k=self.memory_top_k, min_score=self.memory_min_score
            )
        except Exception:
            logger.warning("Memory recall failed; answering without memory", exc_info=True)
            return None
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
        if use_memory and (context := await self._recall(message)):
            sections.append(context)

        messages: list[BaseMessage] = [SystemMessage("\n\n".join(sections))]
        for turn in history:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.content))
        messages.append(HumanMessage(message))
        return messages

    async def _run(
        self, message: str, history: Sequence[ChatTurn], use_memory: bool, use_tools: bool
    ) -> AsyncIterator[str | ToolCallRecord]:
        """Agent loop: yields text chunks as they stream and a record per executed tool call."""
        bound, tools = self._bind_tools(use_tools)
        messages = await self._build_messages(message, history, use_memory, bool(tools))

        first_iteration = 0
        if tools and self.tool_routing:
            # Small models stop calling tools once they start writing text, so for compound
            # messages they answer the easy part and drop the rest. A dedicated routing turn,
            # where answering is not allowed, makes them commit to every needed tool first.
            system = f"{messages[0].content}\n\n{load_prompt('tool_routing')}"
            decision = await bound.ainvoke([SystemMessage(system), *messages[1:]])
            if decision.tool_calls:
                messages.append(AIMessage("", tool_calls=decision.tool_calls))
                tool_messages, records = await execute_tool_calls(decision.tool_calls, tools)
                messages.extend(tool_messages)
                for record in records:
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
            if not tool_calls or not tools or final:
                if tool_calls:
                    logger.warning(
                        "Tool iteration limit reached; ignoring %d call(s)", len(tool_calls)
                    )
                return
            messages.append(response)
            tool_messages, records = await execute_tool_calls(tool_calls, tools)
            messages.extend(tool_messages)
            for record in records:
                yield record

    async def generate(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
    ) -> ChatResult:
        text: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        async for event in self._run(message, history, use_memory, use_tools):
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
        async for event in self._run(message, history, use_memory, use_tools):
            if isinstance(event, str):
                yield event
