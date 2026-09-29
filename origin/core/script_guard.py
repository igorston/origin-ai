"""Repair script slips in streamed replies.

Qwen models sometimes switch into Chinese mid-sentence ("Gandalf, o灰袍巫师, decidiu...":
~1 in 3 stories about Gandalf, and neither the system prompt nor top_p prevented it).
The guard holds back a run of CJK characters as it streams, asks the model to rewrite
just that fragment in the reply's language, and emits the replacement instead. It stays
off when the user writes in those scripts or asks about a language ("como se escreve
obrigado em japonês?"), where such characters are the answer.
"""

import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable

from origin.i18n.packs import rule

logger = logging.getLogger(__name__)

# Kana, CJK ideographs and punctuation, Hangul, fullwidth forms.
_FOREIGN = "　-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯"
FOREIGN_CHAR = re.compile(f"[{_FOREIGN}]")
# A run may contain spaces between foreign characters, never ends in one, and takes the
# spaces before it (so they can go if the run is dropped: "o 巫师, disse" -> "o, disse").
FOREIGN_RUN = re.compile(f" *[{_FOREIGN}](?:[{_FOREIGN}\\s]*[{_FOREIGN}])?")
LANGUAGE_TOPIC = rule("language_topic")
CONTEXT_CHARS = 300
# A replacement much longer than the fragment is the model answering instead of fixing.
MAX_GROWTH = 8

Repair = Callable[[str, str], Awaitable[str]]


def guard_needed(message: str) -> bool:
    """False when foreign scripts are expected in the reply."""
    return not FOREIGN_CHAR.search(message) and not LANGUAGE_TOPIC.search(message)


ARTICLES = {"o", "a", "os", "as", "um", "uma", "el", "la", "los", "las", "un", "the", "an"}
SENTENCE_END = ".!?:\n\"'"


def fit(before: str, replacement: str) -> str:
    """Make a translated fragment read as part of the sentence it lands in."""
    words = replacement.split()
    previous = before.split()[-1:] or [""]
    # "caminharam até o" + "O Monte da Perdição" -> "caminharam até o Monte da Perdição"
    if len(words) > 1 and words[0].lower() in ARTICLES and previous[0].lower() in ARTICLES:
        words = words[1:]
    # Mid-sentence, "Tempestade de poeira" -> "tempestade de poeira"; names with several
    # capitals ("Senhor das Trevas") and single words (maybe a name) are left alone.
    mid_sentence = bool(before.strip()) and before.strip()[-1] not in SENTENCE_END
    capitals = sum(word[:1].isupper() for word in words)
    if mid_sentence and len(words) > 1 and capitals == 1 and words[0][:1].isupper():
        words[0] = words[0][:1].lower() + words[0][1:]
    return " ".join(words)


def _join(before: str, replacement: str, after: str) -> str:
    """CJK uses no spaces, so "o灰袍巫师," must become "o Mago Cinzento,"."""
    replacement = fit(before, replacement)
    if replacement and before[-1:].isalnum() and replacement[0].isalnum():
        replacement = " " + replacement
    if replacement and after[:1].isalnum() and replacement[-1].isalnum():
        replacement += " "
    return replacement


class ScriptGuard:
    def __init__(self, repair: Repair) -> None:
        self.repair = repair
        self.pending = ""
        self.context = ""  # recent emitted text, for the repair prompt
        self.repaired = 0

    def _emit(self, text: str) -> str:
        self.context = (self.context + text)[-CONTEXT_CHARS:]
        return text

    async def _fix(self, run: str, after: str) -> str:
        lead = run[: len(run) - len(run.lstrip(" "))]
        fragment = run[len(lead) :]
        try:
            replacement = (await self.repair(self.context, fragment)).strip().strip("«»\"'")
        except Exception:
            logger.warning("Script repair failed; dropping %r", fragment, exc_info=True)
            replacement = ""
        if FOREIGN_CHAR.search(replacement) or len(replacement) > MAX_GROWTH * len(fragment) + 20:
            logger.warning("Unusable script repair %r for %r; dropping it", replacement, fragment)
            replacement = ""
        self.repaired += 1
        logger.info("Repaired script slip %r -> %r", fragment, replacement)
        if replacement:
            return lead + _join(self.context + lead, replacement, after)
        # Dropped: keep a separating space only if nothing else will separate the words.
        return lead if after[:1].isalnum() else ""

    async def feed(self, chunk: str) -> AsyncIterator[str]:
        self.pending += chunk
        async for text in self._drain(final=False):
            yield text

    async def flush(self) -> AsyncIterator[str]:
        async for text in self._drain(final=True):
            yield text

    async def _drain(self, final: bool) -> AsyncIterator[str]:
        while self.pending:
            match = FOREIGN_RUN.search(self.pending)
            if match is None:
                # Hold trailing spaces back: a foreign run may follow in the next chunk.
                text = self.pending if final else self.pending.rstrip(" ")
                self.pending = self.pending[len(text) :]
                if text:
                    yield self._emit(text)
                return
            if match.start() > 0:
                text, self.pending = self.pending[: match.start()], self.pending[match.start() :]
                yield self._emit(text)
                continue
            if match.end() == len(self.pending) and not final:
                return  # the run may go on in the next chunk
            run, self.pending = self.pending[: match.end()], self.pending[match.end() :]
            replacement = await self._fix(run, self.pending)
            if replacement:
                yield self._emit(replacement)
