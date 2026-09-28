"""Context-window accounting and automatic optimization for chat sessions.

The model only sees `window` tokens. A turn's prompt is: system prompt + tool schemas
(`base`), the conversation summary, the recent messages, recalled memories and the new
message; `reply_reserve` tokens stay free for the answer and tool rounds.

When the prompt passes `compact_at` of the usable budget, the oldest messages are
folded into a running summary until it is back under `compact_target` (the full
transcript stays stored and visible; only the model's view changes). Folding is
sustainable: the summary keeps USER FACTS intact and a code-level guard preserves the
user's own sentences whenever a detail (name, number, code, file...) would be lost.

When the summary outgrows its (soft) budget it is *condensed*: TOPICS shrink, USER FACTS
and PRESERVED never do. That is only attempted while it pays off (TOPICS are a real share
of the summary), at most twice per turn. There is no fixed number of condensations: a
chat of general topics can go on indefinitely, while one full of facts reaches its limit
when what must be preserved, plus the last messages, no longer fits in the window. That
limit follows the machine and the model, like the window itself.
"""

import logging
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from origin.config import Settings
from origin.core.capacity import Capacity, num_ctx
from origin.prompts import load_prompt
from origin.retry import DEFAULT_POLICY, RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)

# Measured on qwen3's tokenizer: 2.4 chars/token for code, ~2.7-3.6 for Portuguese,
# ~5 for English. Estimates use the dense end so they err on the safe side.
CHARS_PER_TOKEN = 2.5
JSON_CHARS_PER_TOKEN = 3.5  # tool schemas
MESSAGE_OVERHEAD = 5  # role markers etc. per message
TOKENS_PER_MEMORY = 40
# The summary may use at most this share of the usable budget, whatever the setting says.
SUMMARY_SHARE = 0.2
# Bounds for calibrating estimates against measurements: English prose measured ~0.5x
# the estimate; dense text (codes, prices, IDs) can measure well above it.
MIN_SCALE = 0.5
MAX_SCALE = 2.0
MIN_CALIBRATION_SAMPLE = 300  # estimated tokens
# Condensing only rewrites TOPICS; below this share of the summary it cannot pay off.
MIN_CONDENSABLE_SHARE = 0.25
# A condensation that shrank the summary less than this is not tried again in the turn.
MIN_CONDENSE_GAIN = 0.1
MAX_CONDENSE_PER_TURN = 2  # bounds the extra latency of one message

State = Literal["ok", "warning", "critical", "closed"]


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD


class Turn(Protocol):
    """A stored or client-sent message (StoredMessage / ChatTurn)."""

    role: str
    content: str


# ---------------------------------------------------------------- budget & usage


class ContextBudget(BaseModel):
    window: int
    reply_reserve: int
    compact_at: float  # fraction of the usable budget that triggers optimization
    compact_target: float  # optimization folds until usage is back under this
    warn_at: float
    critical_at: float
    keep_recent: int
    min_recent: int
    summary_max_tokens: int
    memory_reserve: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContextBudget":
        return cls(
            window=settings.context_window or num_ctx(settings),
            reply_reserve=settings.context_reply_reserve,
            compact_at=settings.context_compact_at,
            compact_target=settings.context_compact_target,
            warn_at=settings.context_warn_at,
            critical_at=0.9,
            keep_recent=settings.context_keep_recent,
            min_recent=settings.context_min_recent,
            summary_max_tokens=settings.context_summary_max_tokens,
            memory_reserve=settings.memory_top_k * TOKENS_PER_MEMORY,
        )

    @property
    def usable(self) -> int:
        return self.window - self.reply_reserve

    @property
    def summary_budget(self) -> int:
        return min(self.summary_max_tokens, int(self.usable * SUMMARY_SHARE))


class ContextUsage(BaseModel):
    window: int
    usable: int  # window minus the reply reserve
    used: int  # prompt tokens (estimated before a turn, measured by Ollama after one)
    percent: float  # used / usable
    state: State
    measured: bool
    base: int
    summary: int
    history: int
    message: int
    memory: int
    compactions: int  # times old messages were folded into the summary (lossless-ish)
    compressions: int  # times the summary's TOPICS were condensed (informational)
    summarized_messages: int
    # USER FACTS + PRESERVED: never condensed, so they are what fills the window for good.
    protected: int = 0
    protected_room: int = 0  # the most the summary and the messages can ever use
    # Where the window size comes from (see origin/core/capacity.py), for the meter.
    window_source: str = "config"
    model: str = ""
    model_limit: int | None = None  # context the model supports
    vram_limit: int | None = None  # ~largest window that fits in VRAM
    vram_total_gb: float | None = None
    gpu: str | None = None


class Compaction(BaseModel):
    folded: int  # messages folded into the summary this time
    compactions: int  # session totals after this step
    compressions: int
    summary_tokens: int
    compressed: bool  # the summary had to be condensed
    preserved: int  # user sentences kept verbatim by the detail guard


class ContextClosed(Exception):
    def __init__(
        self, reason: str, usage: ContextUsage | None = None, partial: "Optimized | None" = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.usage = usage
        self.partial = partial  # optimization done before giving up (worth persisting)


class MessageTooLong(Exception):
    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(f"message of ~{tokens} tokens exceeds the ~{limit} available")
        self.tokens = tokens
        self.limit = limit


# ---------------------------------------------------------------- detail guard

SECTIONS = ("USER FACTS", "TOPICS", "PRESERVED")
DETAIL_PATTERNS = [
    re.compile(r"[\w$€£.,:/-]*\d[\w.,:/-]*"),  # numbers, dates, times, prices, codes
    re.compile(r"\b[\w-]+\.(?:py|js|ts|tsx|md|json|ya?ml|txt|csv|sql|html|css)\b|\b\w+_\w+\b"),
    re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\b"),  # acronyms and all-caps codes
]
CAPITALIZED = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b")


def _fold_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[\W_]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def key_details(text: str) -> set[str]:
    """Specific tokens a summary must not lose: numbers, codes, identifiers, file names
    and proper names (capitalized words that do not start a sentence)."""
    details = {m.group(0).strip(".,:") for p in DETAIL_PATTERNS for m in p.finditer(text)}
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        words = CAPITALIZED.findall(sentence)
        first = sentence.strip().split(" ", 1)[0].strip("\"'(")
        details |= {w for w in words if w != first}
    return {d for d in details if len(_fold_text(d)) >= 2}


# Header keywords, accent-folded, including the translations models sometimes write.
SECTION_KEYWORDS = {
    "USER FACTS": ("fact", "fato", "hecho"),
    "TOPICS": ("topic", "topico", "tema", "assunt"),
    "PRESERVED": ("preserv",),
}


def _section_of(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("#") and stripped.upper() not in SECTION_KEYWORDS:
        return None
    header = _fold_text(stripped)
    if len(header) > 30:  # a sentence, not a header
        return None
    for name, keywords in SECTION_KEYWORDS.items():
        if any(k in header for k in keywords):
            return name
    return None


def split_sections(summary: str) -> dict[str, str]:
    parts = dict.fromkeys(SECTIONS, "")
    current = None
    for line in summary.splitlines():
        if header := _section_of(line):
            current = header
        elif current:
            parts[current] += line + "\n"
    if not any(parts.values()):  # the model ignored the format: treat all as topics
        parts["TOPICS"] = summary
    return {name: text.strip() for name, text in parts.items()}


def protected_tokens(summary: str) -> int:
    """Tokens of the sections that are never condensed (USER FACTS and PRESERVED)."""
    if not summary:
        return 0
    parts = split_sections(summary)
    return sum(estimate_tokens(parts[name]) for name in ("USER FACTS", "PRESERVED") if parts[name])


def worth_condensing(summary: str) -> bool:
    """Condensing rewrites only TOPICS: when they are a small part of the summary (a chat
    full of facts), it would cost a model call and lose detail for almost no room."""
    topics = split_sections(summary)["TOPICS"]
    return bool(topics) and estimate_tokens(topics) >= MIN_CONDENSABLE_SHARE * estimate_tokens(
        summary
    )


def join_sections(parts: dict[str, str]) -> str:
    return "\n\n".join(f"### {name}\n{parts[name] or '-'}" for name in SECTIONS)


def guard_details(sources: Sequence[str], summary: str) -> tuple[str, int]:
    """Re-add, verbatim, every source sentence whose details the summary dropped."""
    parts = split_sections(summary)
    kept = _fold_text(summary)
    preserved = [line for line in parts["PRESERVED"].splitlines() if line.strip("- ")]
    added = 0
    for source in sources:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", source):
            sentence = sentence.strip()
            missing = [d for d in key_details(sentence) if _fold_text(d) not in kept]
            if sentence and missing:
                preserved.append(f"- {sentence}")
                kept += _fold_text(sentence)
                added += 1
    parts["PRESERVED"] = "\n".join(preserved)
    return join_sections(parts), added


HEADING = re.compile(r"^(#{1,2}) +(.+?)\s*$", re.M)
ENDING_CHARS = 240


def piece_outline(text: str) -> str | None:
    """A line recording a story or poem the assistant wrote (writer replies start with a
    heading). Summarizing one, the model kept the plot but lost the title and chapters,
    then invented them when asked, so they are copied from the text itself."""
    if not text.lstrip().startswith("#"):
        return None
    headings = HEADING.findall(text)
    title = next((heading for level, heading in headings if level == "#"), "")
    chapters = [heading for level, heading in headings if level == "##"]
    ending = " ".join(text.split())[-ENDING_CHARS:]
    ending = ending[ending.find(" ") + 1 :] if len(text) > ENDING_CHARS else ending
    line = (
        f'- Piece written by the assistant: "{title}"'
        if title
        else ("- Continuation written by the assistant")
    )
    if chapters:
        line += f"; chapters: {'; '.join(chapters)}"
    return f'{line}; it ends: "...{ending}"'


# ---------------------------------------------------------------- summarizer


MAX_TURN_CHARS = 4000


def _shorten(text: str) -> str:
    """A 2000-word story plus the summary prompt would overflow the summarizer's own
    window (Ollama would then cut the instructions): keep its beginning and end."""
    if len(text) <= MAX_TURN_CHARS:
        return text
    head, tail = MAX_TURN_CHARS * 5 // 8, MAX_TURN_CHARS * 3 // 8
    # Chapter headings of the cut middle stay, so a story's structure survives.
    headings = re.findall(r"^#{1,3} .+$", text[head:-tail], re.M)
    parts = [text[:head], *headings, text[-tail:]]
    return "\n[...]\n".join(parts)


class ConversationSummarizer:
    def __init__(
        self, llm: BaseChatModel, max_tokens: int, retry: RetryPolicy = DEFAULT_POLICY
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.retry = retry

    async def fold(
        self,
        summary: str,
        turns: Sequence[Turn],
        max_tokens: int | None = None,
        *,
        protect: bool = True,
    ) -> tuple[str, int]:
        """Merge `turns` into `summary` (with no turns: condense it). Returns the new
        summary and how many user sentences the detail guard had to preserve.

        Guarded sources: the user's messages, and the previous USER FACTS and PRESERVED
        sections. Assistant explanations are allowed to be condensed. `protect=False`
        (condensing only) lifts the protection: the one lossy step for user facts, used
        only when the facts alone no longer fit (starting a continued conversation)."""
        words = max(40, int((max_tokens or self.max_tokens) * 0.6))
        transcript = "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {_shorten(t.content)}" for t in turns
        )
        if transcript:
            instruction = transcript
        elif protect:
            instruction = "(none — condense the previous summary, TOPICS first)"
        else:
            instruction = (
                "(none — the summary no longer fits. Condense EVERYTHING, USER FACTS included: "
                "merge PRESERVED into USER FACTS, keep the most specific and recent details, "
                "drop the rest, and leave PRESERVED as '-')"
            )
        prompt = load_prompt("conversation_summary").format(
            max_words=words, summary=summary or "(none yet)", messages=instruction
        )
        reply = await call_with_retry(lambda: self.llm.ainvoke(prompt), self.retry, "summary")
        new = split_sections(reply.text.strip())
        if not protect:
            new["PRESERVED"] = ""
            return join_sections(new), 0
        previous = split_sections(summary) if summary else dict.fromkeys(SECTIONS, "")
        # PRESERVED is maintained here, not by the model.
        pieces = [piece_outline(t.content) for t in turns if t.role == "assistant"]
        new["PRESERVED"] = "\n".join(
            line for line in [previous["PRESERVED"], *pieces] if line and line.strip("- ")
        )
        sources = [t.content for t in turns if t.role == "user"]
        sources += [previous["USER FACTS"], previous["PRESERVED"]]
        return guard_details(sources, join_sections(new))


# ---------------------------------------------------------------- manager


@dataclass
class Optimized:
    summary: str
    turns: list
    steps: list[Compaction] = field(default_factory=list)
    compactions: int = 0
    compressions: int = 0
    summarized: int = 0
    scale: float = 1.0  # estimate correction from the last measurement


class ContextManager:
    def __init__(
        self,
        budget: ContextBudget,
        summarizer: ConversationSummarizer | None,
        base_tokens: int,
        capacity: Capacity | None = None,
        model: str = "",
    ) -> None:
        self.budget = budget
        self.summarizer = summarizer
        self.base_tokens = base_tokens  # replaced by a measurement at warmup
        self.capacity = capacity
        self.model = model

    def usage(
        self,
        summary: str,
        turns: Sequence[Turn],
        message: str = "",
        *,
        compactions: int = 0,
        compressions: int = 0,
        summarized: int = 0,
        closed: bool = False,
        measured: int | None = None,
        scale: float = 1.0,
    ) -> ContextUsage:
        """`scale` corrects the text estimates (see `calibrate`)."""
        b = self.budget
        summary_tokens = round(estimate_tokens(summary) * scale) if summary else 0
        counted, estimated = self._split_history(turns)
        history = counted + round(estimated * scale)
        message_tokens = estimate_tokens(message) if message else 0
        memory = b.memory_reserve if message else 0
        estimated = self.base_tokens + summary_tokens + history + message_tokens + memory
        used = measured if measured is not None else estimated
        percent = used / b.usable
        state: State
        if closed:
            state = "closed"
        elif percent >= b.critical_at:
            state = "critical"
        elif percent >= b.warn_at:
            state = "warning"
        else:
            state = "ok"
        return ContextUsage(
            window=b.window,
            usable=b.usable,
            used=used,
            percent=round(percent, 3),
            state=state,
            measured=measured is not None,
            base=self.base_tokens,
            summary=summary_tokens,
            history=history,
            message=message_tokens,
            memory=memory,
            compactions=compactions,
            compressions=compressions,
            protected=round(protected_tokens(summary) * scale),
            protected_room=b.usable - self.base_tokens - b.memory_reserve,
            summarized_messages=summarized,
            model=self.model,
            **(
                {
                    "window_source": self.capacity.source,
                    "model_limit": self.capacity.model_limit,
                    "vram_limit": self.capacity.vram_limit,
                    "vram_total_gb": self.capacity.vram_total_gb,
                    "gpu": self.capacity.gpu,
                }
                if self.capacity
                else {}
            ),
        )

    async def carry_over(self, summary: str, turns: Sequence[Turn]) -> str:
        """Summary for a new conversation that continues this one: everything folded in,
        then condensed until it fits the summary budget, so the new chat starts with room.
        If protected user facts alone overflow, they are condensed too, the only point
        where they may lose detail."""
        if self.summarizer is None:
            return summary
        budget = self.budget.summary_budget
        if turns:
            summary, _ = await self.summarizer.fold(summary, turns, budget)
        if estimate_tokens(summary) > budget:
            summary, _ = await self.summarizer.fold(summary, [], budget // 2)
        for _ in range(2):
            if estimate_tokens(summary) <= budget:
                break
            logger.warning("Continued chat: user facts no longer fit, condensing them too")
            summary, _ = await self.summarizer.fold(summary, [], budget // 2, protect=False)
        return summary

    def check_window(self) -> str | None:
        """Explain why the window is too small for optimization to work well, if it is."""
        b = self.budget
        floor = self.base_tokens + b.memory_reserve + b.summary_budget
        if floor < b.compact_target * b.usable:
            return None
        return (
            f"Context window too small: the fixed prompt ({self.base_tokens}), recalled "
            f"memories ({b.memory_reserve}) and a full summary ({b.summary_budget}) already "
            f"take {floor / b.usable:.0%} of the {b.usable} usable tokens, above the "
            f"{b.compact_target:.0%} optimization target, so long chats will be summarized on "
            "every turn. Raise OLLAMA_NUM_CTX if the models still fit in VRAM."
        )

    @staticmethod
    def _history_tokens(turns: Sequence[Turn]) -> int:
        return sum(getattr(t, "tokens", 0) or estimate_tokens(t.content) for t in turns)

    @staticmethod
    def _split_history(turns: Sequence[Turn]) -> tuple[int, int]:
        """(tokens counted by the model, tokens estimated from text). Assistant replies
        store the model's own count; user messages store an estimate."""
        counted = estimated = 0
        for turn in turns:
            tokens = getattr(turn, "tokens", 0)
            if turn.role == "assistant" and tokens:
                counted += tokens
            else:
                estimated += tokens or estimate_tokens(turn.content)
        return counted, estimated

    def calibrate(self, summary: str, turns: Sequence[Turn], measured: int | None) -> float:
        """Ratio between Ollama's measurement of the last turn (prompt + answer, i.e. about
        the base, the summary and every turn now stored) and our estimate of the same text.
        The estimate assumes ~2.5 chars/token: chat prose is often 1.5-2x lighter (without
        this the chat would compact far too early), while codes and numbers are heavier
        (without this a fact-heavy chat would overflow the reply reserve)."""
        counted, estimated = self._split_history(turns)
        estimated += estimate_tokens(summary) if summary else 0
        # Only estimates are corrected, and only from a sample big enough: after a greeting
        # the recalled memories in the measurement pushed the ratio to its 2x cap, which
        # then doubled a 3455-token story on the meter (shown as ~7000).
        if not measured or estimated < MIN_CALIBRATION_SAMPLE:
            return 1.0
        ratio = (measured - self.base_tokens - counted) / estimated
        return min(MAX_SCALE, max(MIN_SCALE, ratio))

    async def optimize(
        self,
        summary: str,
        turns: list,
        message: str,
        compactions: int = 0,
        compressions: int = 0,
        summarized: int = 0,
        measured: int | None = None,
    ) -> Optimized:
        """Fold old turns into the summary when the prompt passes `compact_at`, down to
        `compact_target`. `measured` is the session's last measured context (calibrates
        the estimates). Raises MessageTooLong if the message alone cannot fit and
        ContextClosed at the operational limit."""
        b = self.budget
        fixed = self.base_tokens + b.memory_reserve
        message_tokens = estimate_tokens(message)
        if fixed + message_tokens > b.usable:
            raise MessageTooLong(message_tokens, b.usable - fixed)

        result = Optimized(summary, list(turns), [], compactions, compressions, summarized)
        result.scale = self.calibrate(summary, turns, measured)

        def current() -> ContextUsage:
            return self.usage(
                result.summary,
                result.turns,
                message,
                compactions=result.compactions,
                compressions=result.compressions,
                summarized=result.summarized,
                scale=result.scale,
            )

        condensed = 0  # this turn
        stalled = False  # a condensation that barely helped: do not repeat it now

        async def condense(force: bool = False) -> tuple[bool, int]:
            """Condense TOPICS if the summary is over budget (or `force`: it does not fit)
            and it can pay off. Returns (condensed, sentences preserved by the guard)."""
            nonlocal condensed, stalled
            if condensed >= MAX_CONDENSE_PER_TURN or stalled:
                return False, 0
            if not (force or over_budget()) or not worth_condensing(result.summary):
                return False, 0
            before = estimate_tokens(result.summary)
            result.summary, preserved = await self.summarizer.fold(
                result.summary, [], b.summary_budget // 2
            )
            result.compressions += 1
            condensed += 1
            gain = 1 - estimate_tokens(result.summary) / before
            if gain < MIN_CONDENSE_GAIN:
                stalled = True
                logger.info("Condensing gained only %.0f%%: not trying again this turn", gain * 100)
            return True, preserved

        def record(folded: int, compressed: bool, preserved: int) -> None:
            result.compactions += 1
            result.summarized += folded
            step = Compaction(
                folded=folded,
                compactions=result.compactions,
                compressions=result.compressions,
                summary_tokens=estimate_tokens(result.summary),
                compressed=compressed,
                preserved=preserved,
            )
            result.steps.append(step)
            logger.info(
                "Context optimized: folded %d messages (compaction %d, condensations %d, "
                "summary ~%d tokens, ~%d protected, %d sentences preserved)",
                folded, step.compactions, step.compressions, step.summary_tokens,
                protected_tokens(result.summary), preserved,
            )  # fmt: skip

        def over_budget() -> bool:
            return estimate_tokens(result.summary) * result.scale > b.summary_budget

        async def fold(turns: list) -> None:
            result.summary, preserved = await self.summarizer.fold(
                result.summary, turns, b.summary_budget
            )
            # The summary budget is soft: when condensing cannot pay off, the summary keeps
            # growing, as long as the conversation still fits in the window.
            compressed, extra = await condense()
            record(len(turns), compressed, preserved + extra)

        if self.summarizer is not None and current().percent >= b.compact_at:
            while current().percent >= b.compact_target and len(result.turns) > b.min_recent:
                keep = b.keep_recent if len(result.turns) > b.keep_recent else b.min_recent
                folded, result.turns = result.turns[:-keep], result.turns[-keep:]
                await fold(folded)

        # MIN_RECENT is a preference, not a floor: when even the last messages do not fit (a
        # long story alone can fill the window), they are folded too rather than closing.
        if self.summarizer is not None and result.turns and current().percent > 1:
            folded, result.turns = result.turns, []
            await fold(folded)

        # Last resort before closing: condense even under budget, while it still helps.
        while self.summarizer is not None and current().percent > 1:
            compressed, preserved = await condense(force=True)
            if not compressed:
                break
            record(0, True, preserved)

        final = current()
        if final.percent > 1:
            if final.protected > final.protected_room / 2:
                reason = (
                    "a conversa atingiu o limite operacional: os fatos e textos preservados "
                    f"dela (~{final.protected} tokens) e as últimas mensagens não cabem mais "
                    f"na janela de contexto ({b.window} tokens)"
                )
            else:
                reason = (
                    "mesmo depois de otimizar, o resumo e as últimas mensagens não cabem na "
                    f"janela de contexto ({b.window} tokens)"
                )
            raise ContextClosed(reason, final, result)
        return result

    def trim(self, turns: list, message: str) -> tuple[list, int]:
        """For client-sent history (nothing to persist a summary in): drop the oldest
        turns until the prompt fits."""
        dropped = 0
        while len(turns) > self.budget.min_recent:
            if self.usage("", turns, message).percent < self.budget.compact_at:
                break
            turns, dropped = turns[1:], dropped + 1
        return turns, dropped
