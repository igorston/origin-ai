import pytest

from origin.core.language import FALLBACK_INSTRUCTION, detect_language, reply_instruction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lembra que meu time favorito é o Sport.", "pt"),
        ("Esquece aquilo da alergia a camarão, era engano.", "pt"),
        ("Que horas são agora?", "pt"),
        ("Please remember that my favorite team is Sport.", "en"),
        ("How many days until Christmas?", "en"),
        ("¿Cuántos días faltan para Navidad?", "es"),
        ("Hola, ¿cómo estás?", "es"),
        ("12345", None),
        ("Docker", None),
    ],
)
def test_detect_language(text: str, expected: str | None) -> None:
    assert detect_language(text) == expected


def test_reply_instruction_is_written_in_the_target_language() -> None:
    assert "português" in reply_instruction("Anota que tenho dentista na quinta.")
    assert "English" in reply_instruction("Remember that I have a dentist on Thursday.")
    assert reply_instruction("OK") == FALLBACK_INSTRUCTION
