import logging
from collections.abc import AsyncIterator, Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from origin.config import Settings
from origin.memory import VectorMemory
from origin.prompts import load_prompt

logger = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMEngine:
    """Thin, model-agnostic wrapper around a LangChain chat model with optional RAG memory."""

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        model_name: str,
        memory: VectorMemory | None = None,
        memory_top_k: int = 4,
        memory_min_score: float = 0.45,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.memory = memory
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score

    @classmethod
    def from_settings(cls, settings: Settings, memory: VectorMemory | None = None) -> "LLMEngine":
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
        )
        return cls(
            model,
            load_prompt("system"),
            settings.ollama_model,
            memory=memory,
            memory_top_k=settings.memory_top_k,
            memory_min_score=settings.memory_min_score,
        )

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
        self, message: str, history: Sequence[ChatTurn] = (), use_memory: bool = True
    ) -> list[BaseMessage]:
        system = self.system_prompt
        if use_memory and (context := await self._recall(message)):
            system = f"{system}\n\n{context}"

        messages: list[BaseMessage] = [SystemMessage(system)]
        for turn in history:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.content))
        messages.append(HumanMessage(message))
        return messages

    async def generate(
        self, message: str, history: Sequence[ChatTurn] = (), use_memory: bool = True
    ) -> str:
        messages = await self._build_messages(message, history, use_memory)
        result = await self.model.ainvoke(messages)
        return result.text

    async def stream(
        self, message: str, history: Sequence[ChatTurn] = (), use_memory: bool = True
    ) -> AsyncIterator[str]:
        messages = await self._build_messages(message, history, use_memory)
        async for chunk in self.model.astream(messages):
            if chunk.text:
                yield chunk.text
