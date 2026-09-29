import json
import re

import pytest

from origin.core.language import detect_language, reply_instruction
from origin.i18n import packs as packs_module
from origin.i18n.packs import LanguagePack, alternatives, packs, rule
from origin.integrations.web import SOURCES_LABEL, needs_live_data

REQUIRED = ("name", "reply")


@pytest.mark.parametrize("code", sorted(packs()))
def test_every_pack_is_complete_and_its_patterns_compile(code: str) -> None:
    data = json.loads((packs_module.PACKS_DIR / f"{code}.json").read_text(encoding="utf-8"))
    assert all(data.get(key) for key in REQUIRED), f"{code}: needs {REQUIRED}"
    pack = packs()[code]
    for key, entries in pack.patterns.items():
        for entry in entries:
            re.compile(entry)  # each alternative on its own, so a bad one is named
            assert "|" not in entry.replace("(", "").split(")")[-1] or "(" in entry, (
                f"{code}.{key}: {entry!r} — top-level '|' belongs in separate entries"
            )


def test_the_rules_are_the_union_of_the_packs() -> None:
    assert rule("live_data").search("Qual a cotação do dólar?")  # pt
    assert rule("live_data").search("What's the weather in Lisbon?")  # en
    assert rule("live_data").search("¿Cuál es el tipo de cambio?")  # es
    assert not rule("live_data").search("Onde eu moro?")


def test_a_rule_no_pack_has_never_matches() -> None:
    assert alternatives("nobody_has_this") == "(?!)"
    assert not rule("nobody_has_this").search("anything at all")


def test_spanish_works_end_to_end() -> None:
    assert detect_language("¿Cuántos días faltan para la Navidad?") == "es"
    assert "español" in reply_instruction("¿Cuántos días faltan para la Navidad?")
    assert SOURCES_LABEL["Spanish"] == "Fuentes"
    assert needs_live_data("¿Qué dice la ley 27.430?")


@pytest.mark.parametrize(
    "message",
    [
        "O que diz a lei 14.790?",
        "E a lei nº 14790?",
        "Como está o PL 2338?",
        "¿Qué dice la ley 27.430?",
    ],
)
def test_law_numbers_of_any_length(message: str) -> None:
    # "\d" before the closing word boundary only matched one-digit numbers.
    assert needs_live_data(message)


def test_a_new_language_is_a_new_file(tmp_path, monkeypatch) -> None:
    for path in packs_module.PACKS_DIR.glob("*.json"):
        (tmp_path / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    french = {
        "name": "French",
        "reply": "[Réponds à l'utilisateur en français, en une ou deux phrases.]",
        "sources_label": "Sources",
        "stopwords": ["le", "la", "est", "je", "vous", "quel", "quelle"],
        "markers": "œ",
        "live_data": ["météo", "cours de l'euro"],
    }
    (tmp_path / "fr.json").write_text(json.dumps(french), encoding="utf-8")
    monkeypatch.setattr(packs_module, "PACKS_DIR", tmp_path)
    packs.cache_clear()
    try:
        assert isinstance(packs()["fr"], LanguagePack)
        assert list(packs())[:3] == ["pt", "en", "es"]  # the evaluated ones keep their order
        assert rule("live_data").search("Quelle est la météo à Paris ?")
        assert rule("live_data").search("Qual a cotação do dólar?")  # the others still count
    finally:
        packs.cache_clear()
