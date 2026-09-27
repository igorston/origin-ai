from langchain_core.messages import AIMessage

from origin.core import LLMEngine
from origin.core.script_guard import ScriptGuard, fit, guard_needed
from tests.fakes import ScriptedChatModel


def fixed(replacement: str):
    calls: list[tuple[str, str]] = []

    async def repair(context: str, fragment: str) -> str:
        calls.append((context, fragment))
        return replacement

    return repair, calls


async def run(guard: ScriptGuard, chunks: list[str]) -> str:
    out = []
    for chunk in chunks:
        out += [text async for text in guard.feed(chunk)]
    out += [text async for text in guard.flush()]
    return "".join(out)


async def test_foreign_run_split_across_chunks_is_repaired_with_spacing() -> None:
    repair, calls = fixed("Mago Cinzento")
    text = await run(ScriptGuard(repair), ["Gandalf, o", "灰袍", "巫师", ", decidiu partir."])
    assert text == "Gandalf, o Mago Cinzento, decidiu partir."
    assert calls == [("Gandalf, o", "灰袍巫师")]  # one repair for the whole run


async def test_clean_text_streams_untouched_and_unbuffered() -> None:
    repair, calls = fixed("x")
    guard = ScriptGuard(repair)
    assert [t async for t in guard.feed("Olá, tudo bem?")] == ["Olá, tudo bem?"]
    assert calls == []


async def test_run_at_the_end_is_repaired_on_flush() -> None:
    repair, _ = fixed("fim")
    assert await run(ScriptGuard(repair), ["E esse foi o ", "结束"]) == "E esse foi o fim"


async def test_unusable_repairs_are_dropped() -> None:
    for bad in ["灰袍巫师", "Claro! Aqui está a tradução: " + "x" * 200]:
        repair, _ = fixed(bad)
        assert await run(ScriptGuard(repair), ["o ", "巫师", ", disse"]) == "o, disse"


async def test_repair_errors_do_not_break_the_reply() -> None:
    async def boom(context: str, fragment: str) -> str:
        raise RuntimeError("down")

    assert await run(ScriptGuard(boom), ["Oi ", "你好", " amigo"]) == "Oi amigo"


def test_translations_are_fitted_into_the_sentence() -> None:
    assert fit("caminharam até o", "O Monte da Perdição") == "Monte da Perdição"
    assert fit("enfrentavam uma", "Tempestade de poeira") == "tempestade de poeira"
    assert fit("diante do", "Senhor das Trevas") == "Senhor das Trevas"
    assert fit("diante de", "Sauron") == "Sauron"
    assert fit("Fim.", "Tempestade de poeira") == "Tempestade de poeira"  # new sentence


def test_guard_stays_off_when_the_script_is_wanted() -> None:
    assert guard_needed("Crie uma história sobre o Gandalf.")
    assert not guard_needed("Como se escreve obrigado em japonês?")
    assert not guard_needed("Traduza 'gato' para chinês")
    assert not guard_needed("O que significa 你好?")


async def test_engine_repairs_streamed_replies() -> None:
    reply = AIMessage("Gandalf, o灰袍巫师, partiu.")
    model = ScriptedChatModel(responses=[reply])
    # Chinese -> English -> the reply's language (detected from the text so far).
    router = ScriptedChatModel(responses=[AIMessage("Grey Wizard"), AIMessage("Mago Cinzento")])
    engine = LLMEngine(model, "sys", "scripted", router=router)

    result = await engine.generate("Conte uma história do Gandalf pra mim.")

    assert result.text == "Gandalf, o Mago Cinzento, partiu."
    first, second = (call[0].content for call in router.received)
    assert "English" in first and "灰袍巫师" in first
    assert "Portuguese" in second and "Grey Wizard" in second


async def test_engine_keeps_requested_scripts() -> None:
    model = ScriptedChatModel(responses=[AIMessage("Obrigado em japonês é ありがとう.")])
    engine = LLMEngine(model, "sys", "scripted")
    result = await engine.generate("Como se diz obrigado em japonês?")
    assert "ありがとう" in result.text
