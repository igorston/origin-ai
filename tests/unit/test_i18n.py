import re
from pathlib import Path

import pytest

from origin.i18n import Translator, catalogs, resolve_locale, translate

STATIC = Path(__file__).parents[2] / "origin" / "web" / "static"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
# t(`prefix.${x}`) in app.js: every value the prefix can take.
DYNAMIC = {
    "ctx.source.": ["vram", "model", "config", "fallback"],
    "ctx.state.": ["ok", "warning", "critical", "closed"],
    "mem.source.": ["agent", "manual"],
    "cal.threshold.": [
        "dedup",
        "conflict",
        "min_score",
        "dedup_hint",
        "conflict_hint",
        "min_score_hint",
    ],
    "cal.score.": [
        "paraphrase",
        "contradiction",
        "compatible",
        "unrelated",
        "relevant",
        "irrelevant",
    ],
}


def placeholders(entry) -> set[str]:
    if isinstance(entry, dict):
        return set().union(*(placeholders(v) for v in entry.values()))
    if isinstance(entry, list):
        return set().union(*(placeholders(v) for v in entry)) if entry else set()
    return set(PLACEHOLDER.findall(entry))


@pytest.mark.parametrize("locale", sorted(set(catalogs()) - {"en"}))
@pytest.mark.parametrize("section", ["server", "web"])
def test_every_catalog_has_the_same_keys_and_placeholders(locale: str, section: str) -> None:
    reference, other = catalogs()["en"][section], catalogs()[locale][section]
    assert set(other) == set(reference), f"{locale} differs: {set(other) ^ set(reference)}"
    for key, entry in reference.items():
        assert type(other[key]) is type(entry), key
        assert placeholders(other[key]) == placeholders(entry), key


def test_every_key_the_interface_uses_exists() -> None:
    html = "".join(
        (STATIC / name).read_text(encoding="utf-8") for name in ("index.html", "login.html")
    )
    js = "".join((STATIC / name).read_text(encoding="utf-8") for name in ("app.js", "login.js"))
    used = set(re.findall(r'data-i18n="([\w.]+)"', html))
    used |= {
        key
        for attr in re.findall(r'data-i18n-attr="([^"]+)"', html)
        for key in re.findall(r":([\w.]+)", attr)
    }
    used |= set(re.findall(r'\bt\("([\w.]+)"', js))
    for prefix in re.findall(r"\bt\(`([\w.]+\.)\$\{", js):
        used |= {prefix + value for value in DYNAMIC[prefix]}
    web = catalogs()["en"]["web"]
    assert used - set(web) == set()
    assert len(used) > 150  # the scan really covered the interface


def test_locale_resolution() -> None:
    assert resolve_locale("pt-BR") == "pt-BR"
    assert resolve_locale("pt") == "pt-BR"
    assert resolve_locale("pt-PT") == "pt-BR"  # same language
    assert resolve_locale("en-US") == "en"
    assert resolve_locale("ja-JP") == "en"  # no catalog: English
    assert resolve_locale(None) == "en"


def test_translate_falls_back_and_pluralizes() -> None:
    assert Translator("pt-BR")("chat.too_long", tokens=10, limit=5).startswith("A mensagem tem ~10")
    assert Translator("en")("chat.too_long", tokens=10, limit=5).startswith("The message has ~10")
    assert translate("pt-BR", "chat.tools", "web", n=1) == "1 tool"
    assert translate("pt-BR", "chat.tools", "web", n=3) == "3 tools"
    assert translate("pt-BR", "no.such.key") == "no.such.key"
