from collections.abc import AsyncIterator, Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from origin.config import Settings
from origin.prompts import load_prompt


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMEngine:
    """Thin, model-agnostic wrapper around a LangChain chat model."""

    def __init__(self, model: BaseChatModel, system_prompt: str, model_name: str) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.model_name = model_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMEngine":
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
        )
        return cls(model, load_prompt("system"), settings.ollama_model)

    def _build_messages(self, message: str, history: Sequence[ChatTurn] = ()) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(self.system_prompt)]
        for turn in history:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.content))
        messages.append(HumanMessage(message))
        return messages

    async def generate(self, message: str, history: Sequence[ChatTurn] = ()) -> str:
        result = await self.model.ainvoke(self._build_messages(message, history))
        return result.text

    async def stream(self, message: str, history: Sequence[ChatTurn] = ()) -> AsyncIterator[str]:
        async for chunk in self.model.astream(self._build_messages(message, history)):
            if chunk.text:
                yield chunk.text
