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
        """Save something about the user to long-term memory.

        Use when the user asks you to remember, note or save something ("lembre", "lembra",
        "guarda", "anota", "não esquece", "remember") — facts, preferences, people, projects,
        appointments, tasks or dates — or shares a stable fact about themselves.
        Never use it for general knowledge, questions, or facts about the world.
        `fact` must be a short, self-contained sentence written from the user's perspective,
        in the user's language (e.g. "Meu time favorito é o Sport.").
        Call it once per distinct fact.
        """
        (memory_id,) = await memory.add([fact], {"source": "agent"})
        return f"Saved to long-term memory (id={memory_id})."

    return [remember]
