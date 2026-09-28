"""Translations for the interface and the server's user-facing messages.

One JSON catalog per locale in this folder (`pt-BR.json`, `en.json`, ...), each with a
`server` and a `web` section. Adding a language is adding a file with the same keys
(tests/unit/test_i18n.py checks that). Plural entries are {"one": ..., "other": ...}.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

CATALOG_DIR = Path(__file__).parent
DEFAULT_LOCALE = "en"


@cache
def catalogs() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CATALOG_DIR.glob("*.json"))
    }


def resolve_locale(requested: str | None) -> str:
    """Best available catalog: exact ("pt-BR"), same language ("pt" -> "pt-BR",
    "en-US" -> "en"), else English."""
    available = catalogs()
    if not requested:
        return DEFAULT_LOCALE
    if requested in available:
        return requested
    language = requested.split("-")[0].lower()
    for name in available:
        if name.split("-")[0].lower() == language:
            return name
    return DEFAULT_LOCALE


def languages() -> dict[str, str]:
    """locale -> its name in its own language, for the language picker."""
    return {name: catalog.get("_name", name) for name, catalog in catalogs().items()}


def translate(locale: str, key: str, section: str = "server", **values: Any) -> str:
    """The message for `key`, falling back to English, then to the key itself."""
    for name in (resolve_locale(locale), DEFAULT_LOCALE):
        entry = catalogs().get(name, {}).get(section, {}).get(key)
        if entry is not None:
            break
    else:
        return key
    if isinstance(entry, dict):  # plural forms
        entry = entry["one"] if values.get("n") == 1 else entry["other"]
    return entry.format(**values) if values else entry


class Translator:
    """Server messages in one locale: `t = Translator("pt-BR"); t("chat.too_long", ...)`."""

    def __init__(self, locale: str | None) -> None:
        self.locale = resolve_locale(locale)

    def __call__(self, key: str, **values: Any) -> str:
        return translate(self.locale, key, **values)


__all__ = ["Translator", "catalogs", "languages", "resolve_locale", "translate"]
