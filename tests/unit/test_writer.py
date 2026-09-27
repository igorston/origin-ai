import pytest
from langchain_core.messages import AIMessage

from origin.core import ChatTurn, LLMEngine
from origin.core.writer import RepetitionFilter, parse_plan, writing_task
from tests.fakes import ScriptedChatModel


@pytest.mark.parametrize(
    ("message", "kind", "length"),
    [
        ("Crie uma história onde o Gandalf vira o senhor do escuro.", "story", "normal"),
        ("Me conta uma historinha curta pra dormir", "story", "short"),
        ("Escreva um conto de terror longo num farol", "story", "long"),
        ("Você pode escrever uma fábula sobre uma raposa?", "story", "normal"),
        ("Faça um poema sobre a saudade", "poem", "normal"),
        ("Escreva um soneto de amor", "poem", "normal"),
        ("Write a short story about a lighthouse", "story", "short"),
        ("Escribe un cuento sobre un gato", "story", "normal"),
    ],
)
def test_creative_requests_are_detected(message: str, kind: str, length: str) -> None:
    task = writing_task(message)
    assert task is not None and (task.kind, task.length) == (kind, length)


@pytest.mark.parametrize(
    "message",
    [
        "Conte a história do Brasil em poucas palavras",  # history, not a story
        "O que é um conto?",
        "Qual a diferença entre um conto e uma crônica?",
        "Lembra que eu gosto de histórias de terror",
        "Continue",  # nothing to continue
    ],
)
def test_other_messages_are_not_writing_tasks(message: str) -> None:
    assert writing_task(message) is None


def test_continuing_a_written_piece() -> None:
    history = [
        ChatTurn(role="user", content="Crie uma história de piratas"),
        ChatTurn(role="assistant", content="# O Mar de Cinzas\n\nEra uma vez..."),
    ]
    task = writing_task("Continue a história, por favor", history)
    assert task is not None and task.continuation.startswith("# O Mar de Cinzas")
    # A normal reply before is not a piece to continue.
    plain = [ChatTurn(role="assistant", content="Claro! Posso ajudar.")]
    assert writing_task("continue", plain) is None


PLAN = """TÍTULO: **O Último Fogo**
PERSONAGENS:
- Gandalf: um mago tentado pelo poder
- Dwalin: um anão desconfiado
CENAS:
1. A Sombra | Gandalf descobre que o Anel sobreviveu. Ele decide procurá-lo.
2. O Pacto | Os anões recusam ajudá-lo. Gandalf os ameaça.
3. O Reino | Gandalf ergue sua torre. O mundo se cala.
"""


def test_plan_parsing_accepts_translated_labels() -> None:
    plan = parse_plan(PLAN)
    assert plan is not None and plan.title == "O Último Fogo"
    assert [title for title, _ in plan.scenes] == ["A Sombra", "O Pacto", "O Reino"]
    assert "Dwalin" in plan.characters
    assert parse_plan("Claro! Aqui está uma história...") is None


def test_repetition_filter_drops_repeats_and_detects_loops() -> None:
    f = RepetitionFilter()
    first = "Ele precisava de um poder que pudesse parar o Senhor das Trevas. Ele sorriu. "
    assert f.feed(first) == first
    # Short sentences may repeat; long ones may not.
    again = f.feed("Ele sorriu. Ele precisava de um poder que pudesse parar o Senhor das Trevas. ")
    assert again == "Ele sorriu. " and not f.looping
    f.feed("A neve caía pesada sobre as colinas de Rohan naquela noite. ")
    f.feed("Ele precisava de um poder que pudesse parar o Senhor das Trevas. ")
    f.feed("A neve caía pesada sobre as colinas de Rohan naquela noite. ")
    assert f.looping


def test_repetition_filter_streams_partial_sentences_on_flush() -> None:
    f = RepetitionFilter()
    assert f.feed("Era uma vez um farol") == ""
    assert f.feed(" no fim do mundo. E") == "Era uma vez um farol no fim do mundo. "
    assert f.flush() == "E"


async def test_story_is_planned_then_written_scene_by_scene() -> None:
    scenes = [
        "Gandalf abriu o pergaminho e leu as palavras antigas em voz baixa.",
        "Os anões recusaram a proposta e fugiram pelas minas escuras.",
        "No alto da torre, Gandalf olhou o mundo silencioso que agora era seu.",
    ]
    model = ScriptedChatModel(responses=[AIMessage(PLAN), *map(AIMessage, scenes)])
    engine = LLMEngine(model, "sys", "scripted")

    result = await engine.generate("Crie uma história onde o Gandalf vira o senhor do escuro.")

    text = result.text
    assert text.startswith("# O Último Fogo\n\n## A Sombra\n\n")
    assert all(scene in text for scene in scenes)
    assert text.index("## O Pacto") < text.index(scenes[1]) < text.index("## O Reino")
    prompts = [call[0].content for call in model.received]
    assert "exactly 5 scenes" in prompts[0]  # normal length
    assert "scene 2 of 3" in prompts[2] and scenes[0] in prompts[2]  # sees the previous scene
    assert "final scene" in prompts[3]
    assert result.tool_calls == []  # the writer does not run the agent loop


async def test_unusable_plans_fall_back_to_a_single_piece() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage("sem formato"),
            AIMessage("ainda sem formato"),
            AIMessage("Era uma vez um farol."),
        ]
    )
    result = await LLMEngine(model, "sys", "scripted").generate("Crie uma história de um farol")
    assert result.text == "Era uma vez um farol."


async def test_poems_are_written_in_one_call_with_a_title() -> None:
    model = ScriptedChatModel(responses=[AIMessage("# Saudade\n\nA casa ficou longe...")])
    result = await LLMEngine(model, "sys", "scripted").generate("Faça um poema curto sobre saudade")
    assert result.text.startswith("# Saudade")
    prompt = model.received[0][0].content
    assert "3 or 4 stanzas" in prompt and "Brazilian Portuguese" in prompt


def test_written_pieces_do_not_count_as_a_context_measurement(client) -> None:
    from main import app
    from origin.api.routes.chat import get_engine

    model = ScriptedChatModel(responses=[AIMessage("# Saudade\n\nA casa ficou longe.")])
    app.dependency_overrides[get_engine] = lambda: LLMEngine(model, "sys", "scripted")
    session_id = client.post("/sessions").json()["id"]
    body = client.post(
        "/chat", json={"message": "Faça um poema sobre saudade", "session_id": session_id}
    ).json()
    assert body["context"]["measured"] is False  # the writer's prompts are not the chat's
    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["context_tokens"] == 0 and detail["messages"][1]["tokens"] > 0
