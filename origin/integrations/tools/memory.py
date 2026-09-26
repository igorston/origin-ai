from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from origin.integrations.registry import ToolContext


def get_tools(ctx: "ToolContext") -> list[BaseTool]:
    if ctx.memory is None:
        return []
    memory = ctx.memory

    @tool
    async def remember(fact: str) -> str:
        """Save a lasting personal fact or preference about the user to long-term memory.

        Use ONLY when the user explicitly asks you to remember something, or shares a stable
        fact about themselves (name, preferences, projects, people, routines).
        Never use it for general knowledge, questions, or facts about the world.
        `fact` must be a short, self-contained sentence written from the user's perspective,
        in the user's language (e.g. "Meu time favorito é o Sport.").
        """
        (memory_id,) = await memory.add([fact], {"source": "agent"})
        return f"Saved to long-term memory (id={memory_id})."

    return [remember]
