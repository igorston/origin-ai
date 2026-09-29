"""Tiny stopword-based language guess, good enough to tell the model which language to use."""

import re

from origin.i18n.packs import packs

# The reply instructions, names, stopwords and marker letters come from the language
# packs (origin/i18n/agent/*.json): a new language is a new pack.
# Instructions are written in the target language itself: an English "reply in
# Portuguese" still primed English words ("Okay, note salva!"). After actions and lookups
# (saving a fact, a date) they ask for a short reply: longer wording made the model pad
# confirmations ("posso ajudar com algo else?"). After web results, the length follows the
# request: "one or two direct sentences" turned "explique a MP das Bets" into two lines.
REPLY_INSTRUCTIONS = {code: pack.reply for code, pack in packs().items()}
DETAILED_INSTRUCTIONS = {code: pack.reply_detailed for code, pack in packs().items()}
FALLBACK_INSTRUCTION = "[Reply in the same language the user wrote in.]"
LANGUAGE_NAMES = {code: pack.name for code, pack in packs().items()}
STOPWORDS = {code: set(pack.stopwords) for code, pack in packs().items()}
# Letters that only one of the candidate languages uses.
MARKERS = {
    code: re.compile(f"[{re.escape(pack.markers)}]")
    for code, pack in packs().items()
    if pack.markers
}


def detect_language(text: str) -> str | None:
    """Return "pt", "en" or "es" for the most likely language, or None if unsure."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    scores = {lang: sum(word in stop for word in words) for lang, stop in STOPWORDS.items()}
    for lang, marker in MARKERS.items():
        if marker.search(lowered):
            scores[lang] += 2
    best = max(scores, key=scores.__getitem__)
    ranked = sorted(scores.values(), reverse=True)
    return best if ranked[0] > 0 and ranked[0] > ranked[1] else None


def reply_instruction(message: str, locale: str | None = None, detailed: bool = False) -> str:
    """Undetected, the configured locale's language (an English "same language as the
    user" after English tool results produced English replies)."""
    language = detect_language(message) or (locale or "")[:2].lower()
    table = DETAILED_INSTRUCTIONS if detailed else REPLY_INSTRUCTIONS
    return table.get(language, FALLBACK_INSTRUCTION)
