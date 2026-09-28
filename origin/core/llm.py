import asyncio
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from origin.branding import get_brand, system_prompt
from origin.config import Settings
from origin.core.agent import (
    ToolCallRecord,
    blocked_for_question,
    execute_tool_calls,
    unbacked_claims,
)
from origin.core.capacity import num_ctx
from origin.core.context import JSON_CHARS_PER_TOKEN, estimate_tokens
from origin.core.language import LANGUAGE_NAMES, detect_language, reply_instruction
from origin.core.script_guard import ScriptGuard, guard_needed
from origin.core.writer import StoryWriter, writing_task
from origin.memory import MemoryHit, VectorMemory
from origin.prompts import load_prompt
from origin.retry import (
    DEFAULT_POLICY,
    RetryPolicy,
    call_with_retry,
    is_mid_stream_failure,
    ollama_client_kwargs,
)

logger = logging.getLogger(__name__)

# How much of the previous exchange is prepended to the memory query for follow-ups.
RECALL_CONTEXT_CHARS = 500


CLAUSE_BOUNDARY = re.compile(r"[?!.;,]+|\s+(?:e|and|y|mas|but)\s+", re.IGNORECASE)


def split_clauses(message: str) -> list[str]:
    """Clauses of a compound message ("X e Y?" -> ["X", "Y"]); [] for a single clause."""
    clauses = [part.strip() for part in CLAUSE_BOUNDARY.split(message)]
    clauses = [part for part in clauses if len(part.split()) >= 2]
    return clauses if len(clauses) > 1 else []


class TurnTimings:
    """Wall-clock time per phase of one agent turn, for latency diagnosis in the logs."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.phases: dict[str, float] = {}
        self.first_token_at: float | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - started

    def first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter() - self.start

    def __str__(self) -> str:
        parts = [f"{name}={seconds * 1000:.0f}ms" for name, seconds in self.phases.items()]
        if self.first_token_at is not None:
            parts.append(f"first_token={self.first_token_at * 1000:.0f}ms")
        parts.append(f"total={(time.perf_counter() - self.start) * 1000:.0f}ms")
        return " ".join(parts)


def make_chat_model(settings: Settings, temperature: float | None = None) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature if temperature is None else temperature,
        reasoning=settings.ollama_reasoning,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=num_ctx(settings),
        **ollama_client_kwargs(RetryPolicy.from_settings(settings)),
    )


def remind_language(tool_messages: list[ToolMessage], message: str) -> list[ToolMessage]:
    """Tool results are English and are the last thing the model reads before replying, which
    made small models answer in English. Append an instruction in the user's own language —
    quoting the user's message instead made the model parrot the quote back as its reply."""
    if tool_messages:
        tool_messages[-1].content += f"\n\n{reply_instruction(message)}"
    return tool_messages


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TurnUsage(BaseModel):
    """Tokens of the final answer call, as counted by the model."""

    input_tokens: int
    output_tokens: int


class ChatResult(BaseModel):
    text: str
    tool_calls: list[ToolCallRecord] = []
    usage: TurnUsage | None = None


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
        retry: RetryPolicy = DEFAULT_POLICY,
        locale: str = "en-US",
    ) -> None:
        self.model = model
        self.locale = locale
        self.retry = retry
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
            system_prompt(get_brand(settings)),
            settings.ollama_model,
            memory=memory,
            memory_top_k=settings.memory_top_k,
            memory_min_score=settings.memory_min_score,
            tools=tools,
            max_tool_iterations=settings.agent_max_tool_iterations,
            tool_routing=settings.agent_tool_routing,
            contextual_recall=settings.memory_contextual_recall,
            router=make_chat_model(settings, temperature=settings.agent_routing_temperature),
            retry=RetryPolicy.from_settings(settings),
            locale=settings.origin_locale,
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
        # Compound messages dilute the embedding ("Onde eu moro e quantos dias faltam pro
        # Natal?" scored 0.44 against "Moro em São Paulo." vs 0.56 alone), so each clause is
        # searched too. Follow-ups like "and her birthday?" carry no entity on their own, so
        # the last exchange is prepended as well. Each memory keeps its best score.
        queries = [message, *split_clauses(message)]
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
        summary: str = "",
    ) -> list[BaseMessage]:
        # The system prompt (and the tool schemas the chat template appends to it) stays
        # identical across calls and turns, so Ollama reuses its KV cache for that prefix.
        # Everything that varies per turn — recalled memories, routing instructions — goes
        # at the end. The conversation summary changes only when the context is optimized.
        sections = [self.system_prompt]
        if with_tools:
            sections.append(load_prompt("tools"))
        if summary:
            sections.append(load_prompt("conversation_summary_context").format(summary=summary))
        messages: list[BaseMessage] = [SystemMessage("\n\n".join(sections))]
        for turn in history:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.content))

        context = await self._recall(message, history) if use_memory else None
        messages.append(HumanMessage(f"{context}\n\n{message}" if context else message))
        return messages

    async def events(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
        summary: str = "",
    ) -> AsyncIterator[str | ToolCallRecord | TurnUsage]:
        """Agent loop: yields text chunks as they stream, a record per executed tool call,
        and finally the token usage of the answer (when the model reports it)."""
        timings = TurnTimings()
        guarded = guard_needed(message)

        async def repair(context: str, fragment: str) -> str:
            return await self._repair_script(context, fragment, message)

        if task := writing_task(message, history):
            # Stories and poems get a writer (plan + scenes) instead of the assistant loop.
            with timings.phase("context"):
                memories = await self._recall(message, history) if use_memory else None
            writer = StoryWriter(
                self.model,
                self.language_of(message),
                guard=lambda: ScriptGuard(repair) if guarded else None,
                retry=self.retry,
            )
            try:
                async for text in writer.write(task, message, memories or ""):
                    timings.first_token()
                    yield text
            finally:
                logger.info("Turn timings: %s | writer=%s", timings, task.kind)
            # The writer's prompts are not the conversation's, so only the output counts
            # (input_tokens=0: the next turn estimates the context instead of trusting it).
            yield TurnUsage(input_tokens=0, output_tokens=writer.output_tokens)
            return

        bound, tools = self._bind_tools(use_tools)
        with timings.phase("context"):
            messages = await self._build_messages(
                message, history, use_memory, bool(tools), summary
            )
        executed: list[ToolCallRecord] = []

        async def run_tools(tool_calls: list[ToolCall]) -> list[ToolCallRecord]:
            with timings.phase("tools"):
                tool_messages, records = await execute_tool_calls(tool_calls, tools, message)
            messages.extend(remind_language(tool_messages, message))
            executed.extend(records)
            return records

        try:
            first_iteration = 0
            if tools and self.tool_routing:
                # Small models stop calling tools once they start writing text, so for compound
                # messages they answer the easy part and drop the rest. A dedicated routing
                # turn, where answering is not allowed, makes them commit to every tool first.
                latest = messages[-1]
                routing = HumanMessage(f"{latest.content}\n\n{load_prompt('tool_routing')}")
                router = self.router.bind_tools(list(tools.values()))
                with timings.phase("routing"):
                    decision = await call_with_retry(
                        lambda: router.ainvoke([*messages[:-1], routing]), self.retry, "routing"
                    )
                calls = [
                    call
                    for call in decision.tool_calls
                    if call["name"] not in tools
                    or not blocked_for_question(tools[call["name"]], message)
                ]
                if calls:
                    messages.append(AIMessage("", tool_calls=calls))
                    for record in await run_tools(calls):
                        yield record
                    first_iteration = 1

            for iteration in range(first_iteration, self.max_tool_iterations + 1):
                # On the last iteration drop the tools so the model is forced to answer.
                final = iteration == self.max_tool_iterations
                model = self.model if final else bound
                for attempt in range(1, self.retry.attempts + 1):
                    response = None
                    emitted = False
                    guard = ScriptGuard(repair) if guarded else None
                    try:
                        async for chunk in model.astream(messages):
                            response = chunk if response is None else response + chunk
                            if chunk.text:
                                emitted = True
                                timings.first_token()
                                if guard is None:
                                    yield chunk.text
                                    continue
                                async for text in guard.feed(chunk.text):
                                    yield text
                        if guard is not None:
                            async for text in guard.flush():
                                yield text
                        break
                    except Exception as exc:
                        # Once text reached the user a retry would splice two different
                        # replies together, so only a failure before the first token
                        # (or during a silent tool-call turn) is retried.
                        last = attempt == self.retry.attempts
                        if emitted or last or not is_mid_stream_failure(exc):
                            raise
                        logger.warning(
                            "Reply stream failed before any text (%s); retry %d/%d",
                            str(exc)[:160], attempt, self.retry.attempts - 1,
                        )  # fmt: skip
                        await asyncio.sleep(self.retry.delay(attempt))

                tool_calls = getattr(response, "tool_calls", None)
                if tool_calls and tools and not final:
                    messages.append(response)
                    for record in await run_tools(tool_calls):
                        yield record
                    continue

                if tool_calls:
                    logger.warning(
                        "Tool iteration limit reached; ignoring %d call(s)", len(tool_calls)
                    )
                if response is not None and tools:
                    with timings.phase("claim_check"):
                        records = await self._verify_claims(
                            tools, messages, response, executed, message
                        )
                    for record in records:
                        yield record
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    yield TurnUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
                return
        finally:
            logger.info("Turn timings: %s | tools=%s", timings, [r.name for r in executed])

    def language_of(self, text: str) -> str:
        """Name of the language to write in; undetected, the configured locale (showing the
        model the text instead made it translate that text)."""
        return LANGUAGE_NAMES.get(detect_language(text) or self.locale[:2], "English")

    async def _repair_script(self, context: str, fragment: str, message: str = "") -> str:
        """Translate a script slip for the ScriptGuard. Qwen translates Chinese into English
        well but into Portuguese poorly ("武士刀" -> "bushinato"; via English: "Katana"), and
        showing it the surrounding text made it repeat the text, so it gets the bare fragment
        and the target language comes from the text itself."""

        async def translate(text: str, language: str) -> str:
            prompt = load_prompt("script_repair").format(fragment=text, language=language)
            reply = await call_with_retry(
                lambda: self.router.ainvoke(prompt), self.retry, "script repair"
            )
            return reply.text.strip()

        # The reply so far may be too short to tell ("Gandalf, o"): the message helps.
        target = self.language_of(f"{message} {context}")
        english = await translate(fragment, "English")
        return english if target == "English" else await translate(english, target)

    async def _verify_claims(
        self,
        tools: Mapping[str, BaseTool],
        messages: list[BaseMessage],
        response: BaseMessage,
        executed: list[ToolCallRecord],
        message: str,
    ) -> list[ToolCallRecord]:
        """If the final reply claims an effect ("anotei!") whose tool was never called, give
        the model one chance to actually call it, so the claim becomes true."""
        claimed = unbacked_claims(response.text, executed, set(tools))
        if not claimed:
            return []
        logger.warning("Reply claims %s without calling it; verifying", sorted(claimed))
        check = HumanMessage(load_prompt("claim_check").format(tools=", ".join(sorted(claimed))))
        router = self.router.bind_tools(list(tools.values()))
        decision = await call_with_retry(
            lambda: router.ainvoke([*messages, AIMessage(response.text), check]),
            self.retry,
            "claim check",
        )
        calls = [call for call in decision.tool_calls if call["name"] in claimed]
        if not calls:
            return []
        _, records = await execute_tool_calls(calls, tools, message)
        return records

    async def generate(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
        summary: str = "",
    ) -> ChatResult:
        text: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        usage = None
        async for event in self.events(message, history, use_memory, use_tools, summary):
            if isinstance(event, str):
                text.append(event)
            elif isinstance(event, TurnUsage):
                usage = event
            else:
                tool_calls.append(event)
        return ChatResult(text="".join(text), tool_calls=tool_calls, usage=usage)

    async def stream(
        self,
        message: str,
        history: Sequence[ChatTurn] = (),
        use_memory: bool = True,
        use_tools: bool = True,
        summary: str = "",
    ) -> AsyncIterator[str]:
        async for event in self.events(message, history, use_memory, use_tools, summary):
            if isinstance(event, str):
                yield event

    # ------------------------------------------------------------ context accounting

    def fixed_prompt(self, with_tools: bool = True) -> list[BaseMessage]:
        """The part of every prompt that does not depend on the conversation."""
        sections = [self.system_prompt]
        if with_tools and self.tools:
            sections.append(load_prompt("tools"))
        return [SystemMessage("\n\n".join(sections))]

    def estimate_base_tokens(self) -> int:
        """Rough size of the fixed prompt, including the tool schemas the chat template
        injects. Replaced by `measure_base_tokens` once the model is available."""
        schemas = json.dumps([convert_to_openai_tool(t) for t in self.tools.values()])
        # JSON tokenizes looser than prose (measured: 802 real vs 1238 at 2.5 chars/token).
        schema_tokens = math.ceil(len(schemas) / JSON_CHARS_PER_TOKEN)
        return estimate_tokens(self.fixed_prompt()[0].content) + schema_tokens

    async def measure_base_tokens(self) -> int:
        """Exact size of the fixed prompt, as counted by the model's own tokenizer."""
        bound, _ = self._bind_tools(True)
        probe = "ok"
        reply = await bound.ainvoke([*self.fixed_prompt(), HumanMessage(probe)])
        measured = (reply.usage_metadata or {}).get("input_tokens", 0)
        return (
            max(0, measured - estimate_tokens(probe)) if measured else self.estimate_base_tokens()
        )
