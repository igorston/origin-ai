"""Long-form creative writing: stories are planned first, then written scene by scene.

Asked for a story in one call, qwen3:8b either wrote a one-paragraph synopsis (under the
assistant prompt) or, told to write 1500+ words, padded the length by looping: the same
paragraphs came back 4-6 times. Short, focused calls do not degenerate, so a story is:

1. a plan (title, characters, N scenes with distinct events), then
2. one streamed call per scene, given the plan and the end of the previous scene,
3. through a repetition filter that drops sentences already written and stops a scene
   that starts looping.

Poems are short enough for a single call with a dedicated prompt.
"""

import logging
import re
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel

from origin.core.context import estimate_tokens
from origin.core.script_guard import ScriptGuard
from origin.i18n.packs import alternatives, packs, rule
from origin.prompts import load_prompt
from origin.retry import DEFAULT_POLICY, RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)

Kind = Literal["story", "poem"]
Length = Literal["short", "normal", "long"]

# --------------------------------------------------------------- request detection

_STORY = alternatives("writer.story")
_POEM = alternatives("writer.poem")
# English "a" is also a Portuguese preposition ("conte a história"): an article only
# after an English verb.
_EN_VERBS = "|".join(packs()["en"].patterns.get("writer.verbs", [])) if "en" in packs() else "(?!)"
_VERBS = alternatives("writer.verbs")
_ARTICLES = alternatives("writer.articles")
# "conte a história do Brasil" is history, "conte uma história" is a story: only
# indefinite articles count (English "a" only after an English verb).
NEW_PIECE = re.compile(
    rf"\b(?P<verb>{_VERBS})\b(?:\s+\w+){{0,3}}?\s+"
    rf"(?P<article>{_ARTICLES})\s+"
    rf"(?:\w+\s+){{0,2}}?(?P<kind>{_STORY}|{_POEM})\b",
    re.IGNORECASE,
)
CONTINUE = rule("writer.continue")
SHORT = rule("writer.short")
LONG = rule("writer.long")


@dataclass
class WritingTask:
    kind: Kind
    length: Length = "normal"
    continuation: str = ""  # the previous piece, when continuing it


class Turn(Protocol):
    role: str
    content: str


def is_piece(text: str) -> bool:
    """A reply written by the writer starts with a heading (a continuation, with its first
    chapter's)."""
    return text.lstrip().startswith("#")


def writing_task(message: str, history: Sequence[Turn] = ()) -> WritingTask | None:
    length: Length = (
        "short" if SHORT.search(message) else "long" if LONG.search(message) else "normal"
    )
    match = NEW_PIECE.search(message)
    if match:
        english_a = match["article"].lower() == "a"
        if english_a and not re.fullmatch(_EN_VERBS, match["verb"], re.I):
            match = None
    if match:
        kind: Kind = "poem" if re.fullmatch(_POEM, match["kind"], re.I) else "story"
        return WritingTask(kind, length)
    last = next((t.content for t in reversed(history) if t.role == "assistant"), "")
    if CONTINUE.search(message) and is_piece(last):
        return WritingTask("story", length, continuation=last)
    return None


# --------------------------------------------------------------- plan

SCENES = {"short": 3, "normal": 5, "long": 8}
WORDS_PER_SCENE = {"short": 300, "normal": 380, "long": 420}
POEM_LENGTH = {
    "short": "3 or 4 stanzas",
    "normal": "5 to 7 stanzas",
    "long": "9 to 12 stanzas",
}
PREVIOUS_CHARS = 1800
# Tokens per scene call: ~2x the target, so a scene can finish but a loop cannot run on.
SCENE_TOKENS = 1100

LABEL = r"^\W*(?:{})\W*:?\s*(.*)$"
TITLE_LINE = re.compile(LABEL.format(alternatives("writer.title_label")), re.I | re.M)
CHARS_LINE = re.compile(LABEL.format(alternatives("writer.characters_label")), re.I | re.M)
SCENES_LINE = re.compile(LABEL.format(alternatives("writer.scenes_label")), re.I | re.M)
SCENE_ITEM = re.compile(r"^\s*\**\s*(\d+)[.)]\s*(.+?)\s*\|\s*(.+?)\s*$", re.M)


@dataclass
class Plan:
    title: str
    characters: str
    scenes: list[tuple[str, str]]  # (scene title, what happens)


def _clean(text: str) -> str:
    return text.strip().strip("*_#\"'«»“” ").strip()


def parse_plan(text: str) -> Plan | None:
    title = TITLE_LINE.search(text)
    scenes = [(_clean(t), _clean(d)) for _, t, d in SCENE_ITEM.findall(text)]
    if not title or not _clean(title[1]) or len(scenes) < 2:
        return None
    characters = ""
    start, end = CHARS_LINE.search(text), SCENES_LINE.search(text)
    if start and end and start.end() < end.start():
        characters = text[start.end() : end.start()].strip()
    return Plan(_clean(title[1]), characters or "-", scenes)


# --------------------------------------------------------------- repetition filter

SENTENCE_END = re.compile(r"(?<=[.!?…])[\"”»')]*[ \t]+|\n+")
MIN_WORDS = 6  # short sentences ("Ele sorriu.") may repeat legitimately
LOOP_AFTER = 2  # consecutive repeated sentences = the model is looping


def _key(sentence: str) -> str:
    return " ".join(re.findall(r"\w+", sentence.lower()))


class RepetitionFilter:
    """Emits text sentence by sentence, dropping sentences already written."""

    def __init__(self, seen: set[str] | None = None) -> None:
        self.seen = seen if seen is not None else set()
        self.buffer = ""
        self.repeats = 0
        self.dropped = 0

    @property
    def looping(self) -> bool:
        return self.repeats >= LOOP_AFTER

    def _take(self, sentence: str) -> str:
        key = _key(sentence)
        if len(key.split()) >= MIN_WORDS and key in self.seen:
            self.repeats += 1
            self.dropped += 1
            # Keep the paragraph break a dropped sentence carried.
            return "\n\n" if sentence.endswith("\n\n") else ""
        if key:
            self.repeats = 0
            self.seen.add(key)
        return sentence

    def feed(self, text: str) -> str:
        self.buffer += text
        out, start = [], 0
        for match in SENTENCE_END.finditer(self.buffer):
            out.append(self._take(self.buffer[start : match.end()]))
            start = match.end()
            if self.looping:
                break
        self.buffer = self.buffer[start:]
        return "".join(out)

    def flush(self) -> str:
        text, self.buffer = self.buffer, ""
        return "" if self.looping else self._take(text)


# --------------------------------------------------------------- writer


class StoryWriter:
    def __init__(
        self,
        llm: BaseChatModel,
        language: str,
        guard: Callable[[], ScriptGuard | None] = lambda: None,
        retry: RetryPolicy = DEFAULT_POLICY,
    ) -> None:
        self.llm = llm
        self.language = language
        self.guard = guard
        self.retry = retry
        # Sampling for prose: a mild penalty over a long window discourages loops.
        self.prose = llm.model_copy(
            update={"num_predict": SCENE_TOKENS, "repeat_penalty": 1.1, "repeat_last_n": 512}
        )
        self.output_tokens = 0  # of the whole piece, for the context accounting

    async def write(self, task: WritingTask, request: str, extra: str = "") -> AsyncIterator[str]:
        if task.kind == "poem":
            async for text in self._poem(task, request, extra):
                yield text
            return
        async for text in self._story(task, request, extra):
            yield text

    async def _stream(self, prompt: str, filtered: RepetitionFilter | None) -> AsyncIterator[str]:
        guard = self.guard()
        generated = ""
        measured = False
        try:
            async with aclosing(self.prose.astream(prompt)) as chunks:
                async for chunk in chunks:
                    if usage := getattr(chunk, "usage_metadata", None):
                        self.output_tokens += usage.get("output_tokens", 0)
                        measured = True
                    if not chunk.text:
                        continue
                    generated += chunk.text
                    pieces = [t async for t in guard.feed(chunk.text)] if guard else [chunk.text]
                    for piece in pieces:
                        text = filtered.feed(piece) if filtered else piece
                        if text:
                            yield text
                    if filtered and filtered.looping:
                        logger.warning("Scene started repeating itself; ending it there")
                        return
        finally:
            if not measured:  # cut short (loop, disconnect) before Ollama reported usage
                self.output_tokens += estimate_tokens(generated) if generated else 0
        tail = "".join([t async for t in guard.flush()]) if guard else ""
        if filtered:
            tail = filtered.feed(tail) + filtered.flush()
        if tail:
            yield tail

    async def _poem(self, task: WritingTask, request: str, extra: str) -> AsyncIterator[str]:
        prompt = load_prompt("poem").format(
            request=request,
            extra=extra,
            language=self.language,
            length=POEM_LENGTH[task.length],
        )
        async for text in self._stream(prompt, None):
            yield text

    async def _plan(self, task: WritingTask, request: str, extra: str) -> Plan | None:
        if task.continuation:
            excerpt = task.continuation[:1500] + "\n[...]\n" + task.continuation[-2500:]
            extra = (
                f"{extra}\nThis continues an existing story. Plan only what happens NEXT; do "
                f"not retell it. The story so far (beginning and end):\n{excerpt}\n"
            )
        prompt = load_prompt("story_plan").format(
            request=request, extra=extra, language=self.language, scenes=SCENES[task.length]
        )
        for attempt in range(2):
            reply = await call_with_retry(lambda: self.llm.ainvoke(prompt), self.retry, "plan")
            plan = parse_plan(reply.text)
            if plan:
                return plan
            logger.warning("Unusable story plan (attempt %d): %r", attempt + 1, reply.text[:300])
        return None

    async def _story(self, task: WritingTask, request: str, extra: str) -> AsyncIterator[str]:
        plan = await self._plan(task, request, extra)
        if plan is None:  # write it as a single scene rather than fail
            plan = Plan("", "-", [("", request)])
        if not task.continuation and plan.title:
            yield f"# {plan.title}\n\n"
        outline = "\n".join(f"{i}. {t}: {d}" for i, (t, d) in enumerate(plan.scenes, 1))
        story = task.continuation
        seen: set[str] = set()
        filtered = RepetitionFilter(seen)
        for sentence in re.split(r"(?<=[.!?])\s+", story):
            seen.add(_key(sentence))
        total = len(plan.scenes)
        for number, (scene_title, scene) in enumerate(plan.scenes, 1):
            if total > 1 and scene_title:
                heading = f"## {scene_title}\n\n"
                story += heading
                yield heading
            if number == 1 and not task.continuation:
                position = (
                    "This is the opening: introduce the protagonist and the world through action."
                )
            elif number == total:
                position = (
                    "This is the final scene: continue from where the previous one ended, bring "
                    "the climax to its resolution and close the story with a final image."
                )
            else:
                position = "Continue directly from where the previous scene ended."
            previous = (
                f'\nThe story so far ends like this:\n"...{story[-PREVIOUS_CHARS:].strip()}"\n'
                if story.strip()
                else ""
            )
            prompt = load_prompt("story_scene").format(
                language=self.language,
                title=plan.title or request,
                request=request,
                characters=plan.characters,
                outline=outline,
                previous=previous,
                number=number,
                total=total,
                scene_title=scene_title,
                scene=scene,
                words=WORDS_PER_SCENE[task.length],
                position=position,
            )
            filtered.repeats = 0
            async for text in self._stream(prompt, filtered):
                story += text
                yield text
            ending = "" if number == total else "\n\n"
            story = story.rstrip() + "\n\n"
            if ending:
                yield ending
        if filtered.dropped:
            logger.info("Story: dropped %d repeated sentences", filtered.dropped)
