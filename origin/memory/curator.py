"""Memory curation: normalize, deduplicate and supersede facts before they are stored.

Shared by the `remember` tool, manual adds and edits, so every path that writes a
memory gets the same hygiene.
"""

import asyncio
import logging
import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, timedelta

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from origin.core.language import detect_language
from origin.i18n.packs import packs, rule
from origin.memory.vectorstore import MemoryHit, MemoryRecord, MetadataValue, VectorMemory
from origin.prompts import load_prompt
from origin.retry import DEFAULT_POLICY, RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)

MAX_FACTS = 10

# Words that may legitimately disappear when a note is rewritten ("eu", "agora", ...).
# Accent-folded, since comparison happens on folded text.
# The change words ("mudei", "moved") are among them: the rules say to keep the current
# state ("mudei de emprego, agora trabalho na X" -> "Trabalho na X."), so the transition
# itself may be dropped. From the language packs' filler_words.
STOPWORDS = frozenset().union(*(pack.filler_words for pack in packs().values()))

LANGUAGE_NAMES = {code: pack.short_name for code, pack in packs().items()}

# "Agora também gosto de X" adds to what is known; the judge sometimes read it as a change.
ADDITIVE = rule("additive")
# The judge names what each sentence is about before answering: "A: ... | B: ... | NO".
VERDICT = re.compile(r"\b(YES|NO)\b")


class CurationResult(BaseModel):
    saved: list[MemoryRecord] = []  # memories created or updated
    duplicates: list[MemoryRecord] = []  # existing memories that already said it
    archived: list[MemoryRecord] = []  # memories superseded by the new facts
    normalized: bool = False  # the text was rewritten by the model


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", _fold(text))
    return {w for w in words if (len(w) >= 3 and not w.isdigit()) and w not in STOPWORDS}


def missing_words(note: str, facts: list[str]) -> set[str]:
    """Content words of the note absent from the rewritten facts (prefix match, so
    'alergico' is covered by 'alérgico' and 'trabalho' by 'trabalha')."""
    written = _content_words(" ".join(facts))
    # Also match across spacing changes ("vscode" -> "VS Code").
    compact = re.sub(r"[\W_]+", "", _fold(" ".join(facts)))
    return {word for word in _content_words(note) if not _matches(word, written, compact)}


def _matches(word: str, pool: set[str], compact: str) -> bool:
    return word in compact or any(w.startswith(word[:5]) or word.startswith(w[:5]) for w in pool)


def added_words(note: str, facts: list[str]) -> set[str]:
    """Content words in the facts that the note never mentions. A rewrite may add a
    connective or two ("se chama", "dois" for "2"); more means invented facts (seen:
    "My sister Julia..." -> "Tenho duas irmãs, Julia e Ana")."""
    source = _content_words(note)
    compact = re.sub(r"[\W_]+", "", _fold(note))
    return {w for w in _content_words(" ".join(facts)) if not _matches(w, source, compact)}


# Connectives a clean rewrite commonly adds; they never count as invented.
ADDABLE = {
    "chama",
    "chamo",
    "chamada",
    "chamado",
    "called",
    "named",
    "llama",
    "dois",
    "duas",
    "two",
}
MAX_ADDED_WORDS = 1

# A stored "amanhã" is wrong the next day, so relative days become absolute dates before
# the model sees the note. Done in code: the model ignored a supplied calendar, and small
# models get calendar arithmetic wrong. Longest phrases first.
RELATIVE_DAYS = [
    (r"depois de amanh[ãa]", 2, "em", "%d/%m/%Y"),
    (r"anteontem", -2, "em", "%d/%m/%Y"),
    (r"amanh[ãa]", 1, "em", "%d/%m/%Y"),
    (r"ontem", -1, "em", "%d/%m/%Y"),
    (r"hoje", 0, "em", "%d/%m/%Y"),
    (r"the day after tomorrow", 2, "on", "%Y-%m-%d"),
    (r"tomorrow", 1, "on", "%Y-%m-%d"),
    (r"yesterday", -1, "on", "%Y-%m-%d"),
    (r"today", 0, "on", "%Y-%m-%d"),
    (r"pasado mañana", 2, "el", "%d/%m/%Y"),
    # "por la mañana" / "esta mañana" mean "in the morning", not "tomorrow".
    (r"(?<!la )(?<!esta )mañana", 1, "el", "%d/%m/%Y"),
    (r"ayer", -1, "el", "%d/%m/%Y"),
]


def resolve_relative_days(text: str, today: date | None = None) -> str:
    today = today or datetime.now().astimezone().date()
    for pattern, offset, preposition, fmt in RELATIVE_DAYS:
        day = (today + timedelta(days=offset)).strftime(fmt)
        text = re.sub(rf"\b{pattern}\b", f"{preposition} {day}", text, flags=re.IGNORECASE)
    return text


def parse_facts(output: str) -> list[str]:
    facts = []
    for line in output.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if line and not line.lower().startswith(("facts:", "note:")):
            facts.append(line)
    return facts[:MAX_FACTS]


class MemoryCurator:
    def __init__(
        self,
        memory: VectorMemory,
        llm: BaseChatModel | None,
        conflict_threshold: float = 0.55,
        conflict_candidates: int = 3,
        retry: RetryPolicy = DEFAULT_POLICY,
    ) -> None:
        self.retry = retry
        self.memory = memory
        self.llm = llm
        self.conflict_threshold = conflict_threshold
        self.conflict_candidates = conflict_candidates

    # ------------------------------------------------------------ model calls

    async def normalize(self, note: str) -> list[str]:
        """Rewrite a free-form note into clean atomic facts. Guarded: if the rewrite loses
        content words (dropped details, translation), retry once naming them; if it still
        does, fall back to the original note. User data is never lost to a bad rewrite."""
        note = resolve_relative_days(note.strip())
        if self.llm is None or not note:
            return [note]
        prompt = load_prompt("memory_normalize").format(text=note)
        # The examples are mostly Portuguese; without naming the language the model
        # translated English notes despite the rule.
        if language := LANGUAGE_NAMES.get(detect_language(note) or ""):
            head, _, tail = prompt.rpartition("Facts:")  # only the answer slot, not examples
            prompt = f"{head}Facts (in {language}, like the note):{tail}"
        for attempt in range(2):
            try:
                reply = await call_with_retry(
                    lambda p=prompt: self.llm.ainvoke(p), self.retry, "memory normalization"
                )
                facts = parse_facts(reply.text)
            except Exception:
                logger.warning("Memory normalization failed; storing as written", exc_info=True)
                return [note]
            lost = missing_words(note, facts)
            invented = added_words(note, facts) - ADDABLE
            if facts and not lost and len(invented) <= MAX_ADDED_WORDS:
                return facts
            logger.info(
                "Normalization attempt %d rejected (lost=%s, added=%s): %s",
                attempt + 1, sorted(lost), sorted(invented), facts,
            )  # fmt: skip
            problems = []
            if lost:
                problems.append(f"dropped or translated: {', '.join(sorted(lost))}")
            if len(invented) > MAX_ADDED_WORDS:
                problems.append(f"added words not in the note: {', '.join(sorted(invented))}")
            prompt += (
                f"\n(Your previous answer {facts!r} {'; '.join(problems)}. Use only the note's "
                "information, in the note's language.)\n"
            )
        return [note]

    async def supersedes(self, old: str, new: str) -> bool:
        if self.llm is None or ADDITIVE.search(new):
            return False
        prompt = load_prompt("memory_conflict").format(old=old, new=new)
        try:
            reply = await call_with_retry(
                lambda: self.llm.ainvoke(prompt), self.retry, "memory conflict check"
            )
            verdicts = VERDICT.findall(reply.text.upper())
        except Exception:
            logger.warning("Memory conflict check failed; keeping %r", old, exc_info=True)
            return False
        # Naming both subjects first made qwen3:8b see that "torço pro Náutico" and "meu time
        # favorito é o Sport" are the same thing, and "gosto de rock / de samba" are not.
        # The prompt asks whether both can be true at once: "NO" means the new fact replaces.
        return bool(verdicts) and verdicts[-1] == "NO"

    # ------------------------------------------------------------ storage

    async def _archive_superseded(self, fact: str, new_id: str) -> list[MemoryRecord]:
        similar: list[MemoryHit] = await self.memory.search(
            fact, k=self.conflict_candidates + 1, min_score=self.conflict_threshold
        )
        similar = [hit for hit in similar if hit.id != new_id][: self.conflict_candidates]
        # Similarity alone cannot tell "Sport -> Náutico" (replace) from "pizza de calabresa /
        # pizza de mussarela" (both true), so a focused yes/no call decides, in parallel.
        # Superseded facts are archived, not deleted, so a wrong call is recoverable.
        verdicts = await asyncio.gather(*(self.supersedes(hit.content, fact) for hit in similar))
        archived = []
        for hit, supersedes in zip(similar, verdicts, strict=True):
            if supersedes:
                self.memory.archive(hit.id, superseded_by=new_id)
                archived.append(self.memory.get(hit.id))
        return archived

    async def save_fact(
        self, fact: str, metadata: Mapping[str, MetadataValue], result: CurationResult
    ) -> None:
        """Store one already-clean fact: skip duplicates, archive what it supersedes."""
        if duplicate := await self.memory.find_duplicate(fact):
            result.duplicates.append(self.memory.get(duplicate.id))
            return
        (memory_id,) = await self.memory.add([fact], metadata, dedup=False)
        result.archived += await self._archive_superseded(fact, memory_id)
        result.saved.append(self.memory.get(memory_id))

    async def add(
        self, note: str, metadata: Mapping[str, MetadataValue], optimize: bool = True
    ) -> CurationResult:
        facts = await self.normalize(note) if optimize else [note.strip()]
        result = CurationResult(normalized=facts != [note.strip()])
        for fact in facts:
            await self.save_fact(fact, metadata, result)
        return result

    async def edit(self, memory_id: str, note: str, optimize: bool = True) -> CurationResult:
        """Replace a memory with the (normalized) note. The first fact rewrites the memory
        in place; extra facts become new memories with the same source."""
        record = self.memory.get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        facts = await self.normalize(note) if optimize else [note.strip()]
        result = CurationResult(normalized=facts != [note.strip()])
        first, *rest = facts

        duplicate = await self.memory.find_duplicate(first)
        if duplicate and duplicate.id != memory_id:
            # The edit now says what another memory already says: merge into that one.
            self.memory.archive(memory_id, superseded_by=duplicate.id)
            result.archived.append(self.memory.get(memory_id))
            result.duplicates.append(self.memory.get(duplicate.id))
        else:
            await self.memory.update(memory_id, first)
            result.archived += await self._archive_superseded(first, memory_id)
            result.saved.append(self.memory.get(memory_id))

        metadata = {"source": record.metadata.get("source", "manual")}
        for fact in rest:
            await self.save_fact(fact, metadata, result)
        return result
