import re
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from origin.memory.curator import CurationResult, MemoryCurator
from origin.retry import RetryPolicy

if TYPE_CHECKING:
    from origin.integrations.registry import ToolContext

MEMORY_ID = re.compile(r"^[0-9a-f]{32}$")
# A description must match a stored memory at least this well to be forgotten.
FORGET_MIN_SCORE = 0.6
EXPLICIT_SAVE_INTENT = (
    r"\b(lembr(e|a|ar)\s+(de\s+)?que|anot|guard|salv|registr|n[ãa]o\s+esque[cç]|"
    r"remember\s+that|note\s+that|save)"
)


def get_tools(ctx: "ToolContext") -> list[BaseTool]:
    if ctx.memory is None:
        return []
    store = ctx.memory
    curator = ctx.curator or MemoryCurator(
        store,
        ctx.llm,
        ctx.settings.memory_conflict_threshold,
        retry=RetryPolicy.from_settings(ctx.settings),
    )

    @tool
    async def remember(fact: str) -> str:
        """Save something about the user to long-term memory.

        Use when the user asks you to remember, note or save something ("lembre", "lembra",
        "guarda", "anota", "não esquece", "remember") — facts, preferences, people, projects,
        appointments, tasks or dates. "Lembra que <fato>" is a request to save the fact (an
        imperative), even if it reads like "do you remember?" — save it unless the exact
        fact is already listed in the memories provided with the message.
        Also use it when the user shares a stable fact about themselves, including changes
        ("mudei de...", "agora eu...").
        Never use it for questions or requests for advice (answer those instead), general
        knowledge, or facts about the world. "Lembra que <fato>" is not a question.
        `fact` must be a short, self-contained sentence stating the CURRENT fact from the
        user's perspective, in the user's language: "Meu time favorito é o Náutico." — not
        "Mudei de time, agora torço pro Náutico."; "Moro em São Paulo." — not "Me mudei...".
        Call it once per distinct fact.
        """
        # The agent already writes a clean fact, so no normalization pass here (latency);
        # dedup and superseding are shared with manual adds and edits via the curator.
        result = CurationResult()
        await curator.save_fact(fact, {"source": "agent"}, result)
        if result.duplicates:
            (duplicate,) = result.duplicates
            return f'Already in memory (id={duplicate.id}): "{duplicate.content}".'

        (saved,) = result.saved
        output = f"Saved to long-term memory (id={saved.id})."
        if result.archived:
            listing = "\n".join(f'- "{record.content}"' for record in result.archived)
            output += f"\nIt replaces these outdated memories, which were archived:\n{listing}"
        # The fact is stored in the user's first person; without this, small models often
        # confirmed by repeating it verbatim ("Entendi. Moro em Recife.").
        output += (
            "\nWhen confirming, rephrase the fact in the second person, speaking to the user "
            "(the fact above is in the user's own words)."
        )
        return output

    # Pure questions ("Onde eu moro?", "O que levo de presente pra ela?") never save, unless
    # they carry an explicit save request ("Você pode anotar que...?", "Lembra que ...?").
    # "Você lembra o nome dela?" asks to recall, so bare "lembra" does not count.
    remember.metadata = {"not_for_questions": True, "explicit_intent": EXPLICIT_SAVE_INTENT}

    @tool
    async def forget(memory: str) -> str:
        """Delete a memory that is wrong or outdated.

        Use when the user asks you to forget or delete something ("esquece", "apaga",
        "era engano"). Do not use it for changes like "mudei de time": `remember` the new
        fact instead, and outdated memories are replaced automatically.
        `memory` is either a memory id (from a previous tool result) or a short description
        of the fact to forget (e.g. "alergia a camarão").
        """
        target_id = memory.strip()
        if MEMORY_ID.match(target_id):
            record = store.get(target_id)
            if record is None:
                return f"No memory with id={target_id}."
            content = record.content
        else:
            hits = await store.search(memory, k=1, min_score=FORGET_MIN_SCORE)
            if not hits:
                return f'No stored memory matches "{memory}"; nothing was deleted.'
            target_id, content = hits[0].id, hits[0].content
        await store.delete([target_id])
        return f'Deleted memory (id={target_id}): "{content}".'

    return [remember, forget]
