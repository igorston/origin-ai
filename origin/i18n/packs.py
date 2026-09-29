"""Language packs: what the agent knows about each language, one file per language in
`origin/i18n/agent/<code>.json` (a 2-letter code: pt, en, es).

The interface catalogs (`origin/i18n/<locale>.json`) translate the screens; a pack holds
what the agent needs to behave in that language: how to tell it apart, the reply
instructions written in it, and the phrases that trigger behavior (a live-data question
gets a web search, "também gosto de" adds instead of replacing, "escreva uma história"
goes to the story writer). Adding a language is adding its pack; every rule is the union
of all packs, so a message is understood whatever language the interface is in.

Pattern entries are regular-expression alternatives, joined into one expression per rule.
"""

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

PACKS_DIR = Path(__file__).parent / "agent"


@dataclass(frozen=True)
class LanguagePack:
    code: str
    name: str  # as the model is told: "Brazilian Portuguese"
    short_name: str  # in the memory normalization hint: "Portuguese"
    reply: str  # after actions and lookups: short
    reply_detailed: str  # after web results: as long as the request needs
    sources_label: str = "Sources"
    stopwords: frozenset[str] = frozenset()  # frequent words, to detect the language
    markers: str = ""  # letters only this language uses among the packs (ã, ñ)
    filler_words: frozenset[str] = frozenset()  # may vanish when a note is rewritten
    weekdays: tuple[str, ...] = ()  # Monday first, as the date tools write them
    patterns: dict[str, list[str]] = field(default_factory=dict)


def _load(path: Path) -> LanguagePack:
    data = json.loads(path.read_text(encoding="utf-8"))
    patterns = {
        key: value for key, value in data.items() if isinstance(value, list) and key not in
        ("stopwords", "filler_words", "weekdays")
    }  # fmt: skip
    patterns.update({f"writer.{key}": value for key, value in data.get("writer", {}).items()})
    return LanguagePack(
        code=path.stem,
        name=data["name"],
        short_name=data.get("short_name", data["name"]),
        reply=data["reply"],
        reply_detailed=data.get("reply_detailed", data["reply"]),
        sources_label=data.get("sources_label", "Sources"),
        stopwords=frozenset(data.get("stopwords", [])),
        markers=data.get("markers", ""),
        filler_words=frozenset(data.get("filler_words", [])),
        weekdays=tuple(data.get("weekdays", [])),
        patterns=patterns,
    )


# Portuguese first: the language the project was built and evaluated in.
ORDER = {"pt": 0, "en": 1, "es": 2}


@cache
def packs() -> dict[str, LanguagePack]:
    found = sorted(PACKS_DIR.glob("*.json"), key=lambda p: (ORDER.get(p.stem, 99), p.stem))
    return {path.stem: _load(path) for path in found}


def alternatives(key: str) -> str:
    """Every pack's alternatives for `key`, as one regex alternation (no group)."""
    parts = [alt for pack in packs().values() for alt in pack.patterns.get(key, [])]
    return "|".join(parts) or r"(?!)"  # (?!) never matches: no pack has the rule


def rule(key: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """`\\b(alternatives)\\b`, the shape of every phrase rule."""
    return re.compile(rf"\b({alternatives(key)})\b", flags)


def by_name(name: str) -> LanguagePack | None:
    return next((pack for pack in packs().values() if pack.name == name), None)
