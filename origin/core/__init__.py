"""The assistant's core. The names below load on first use: importing a light module of
this package (origin.core.language, used by the memory curator) must not import the
whole engine, which imports the curator back (a plugin importing origin.memory.curator
first failed on that cycle)."""

from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "ToolCallRecord": "origin.core.agent",
    "ChatResult": "origin.core.llm",
    "ChatTurn": "origin.core.llm",
    "LLMEngine": "origin.core.llm",
    "TurnUsage": "origin.core.llm",
    "make_chat_model": "origin.core.llm",
}

__all__ = ["ChatResult", "ChatTurn", "LLMEngine", "ToolCallRecord", "TurnUsage", "make_chat_model"]

if TYPE_CHECKING:
    from origin.core.agent import ToolCallRecord
    from origin.core.llm import ChatResult, ChatTurn, LLMEngine, TurnUsage, make_chat_model


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'origin.core' has no attribute {name!r}")
