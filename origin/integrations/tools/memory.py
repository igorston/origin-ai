import asyncio
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from origin.prompts import load_prompt

if TYPE_CHECKING:
    from origin.integrations.registry import ToolContext

logger = logging.getLogger(__name__)

MEMORY_ID = re.compile(r"^[0-9a-f]{32}$")
# A description must match a stored memory at least this well to be forgotten.
FORGET_MIN_SCORE = 0.6
# Most similar stored memories checked for being superseded by a new fact.
CONFLICT_CANDIDATES = 3
EXPLICIT_SAVE_INTENT = (
    r"\b(lembr(e|a|ar)\s+(de\s+)?que|anot|guard|salv|registr|n[ãa]o\s+esque[cç]|"
    r"remember\s+that|note\s+that|save)"
)


def get_tools(ctx: "ToolContext") -> list[BaseTool]:
    if ctx.memory is None:
        return []
    store = ctx.memory
    conflict_threshold = ctx.settings.memory_conflict_threshold

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
        if duplicate := await store.find_duplicate(fact):
            return f'Already in memory (id={duplicate.id}): "{duplicate.content}".'

        similar = await store.search(fact, k=CONFLICT_CANDIDATES, min_score=conflict_threshold)
        (memory_id,) = await store.add([fact], {"source": "agent"}, dedup=False)
        output = f"Saved to long-term memory (id={memory_id})."

        # Similarity alone cannot tell "Sport -> Náutico" (replace) from "pizza de calabresa /
        # pizza de mussarela" (both true), so a focused yes/no call decides, one per candidate,
        # in parallel. Superseded facts are archived, not deleted, so a wrong call is
        # recoverable.
        verdicts = await asyncio.gather(*(_supersedes(hit.content, fact) for hit in similar))
        replaced = [hit for hit, supersedes in zip(similar, verdicts, strict=True) if supersedes]
        for hit in replaced:
            store.archive(hit.id, superseded_by=memory_id)
        if replaced:
            listing = "\n".join(f'- "{hit.content}"' for hit in replaced)
            output += f"\nIt replaces these outdated memories, which were archived:\n{listing}"
        # The fact is stored in the user's first person; without this, small models often
        # confirmed by repeating it verbatim ("Entendi. Moro em Recife.").
        output += (
            "\nWhen confirming, rephrase the fact in the second person, speaking to the user "
            "(the fact above is in the user's own words)."
        )
        return output

    async def _supersedes(old: str, new: str) -> bool:
        if ctx.llm is None:
            return False
        prompt = load_prompt("memory_conflict").format(old=old, new=new)
        try:
            answer = (await ctx.llm.ainvoke(prompt)).text.strip().upper()
        except Exception:
            logger.warning("Memory conflict check failed; keeping %r", old, exc_info=True)
            return False
        # The prompt asks whether both can be true at once: "NO" means the new fact replaces.
        return answer.startswith("NO")

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
