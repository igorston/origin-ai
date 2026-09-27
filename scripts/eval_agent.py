"""Agent eval: tool routing and answer quality against the real local models.

Usage:
    python scripts/eval_agent.py                       # current settings, 3 runs per case
    python scripts/eval_agent.py --runs 5 --temperature 0 --reasoning true
    python scripts/eval_agent.py --only compound

Each case runs against a fresh, isolated in-memory Chroma collection.
"""

import argparse
import asyncio
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

from origin.config import get_settings  # noqa: E402
from origin.core import ChatResult, ChatTurn, LLMEngine, make_chat_model  # noqa: E402
from origin.integrations import ToolContext, ToolRegistry  # noqa: E402
from origin.memory import VectorMemory  # noqa: E402

TODAY = datetime.now().astimezone().date()


def days_to(month: int, day: int) -> int:
    target = TODAY.replace(month=month, day=day)
    if target < TODAY:
        target = target.replace(year=TODAY.year + 1)
    return (target - TODAY).days


DAYS_TO_XMAS = days_to(12, 25)
DAYS_TO_NEW_YEAR = days_to(1, 1)
WEEKDAY_PT = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"][TODAY.weekday()]

Check = Callable[[ChatResult, list[str]], bool]


def says(*needles: str) -> Check:
    return lambda r, _: all(n.lower() in r.text.lower() for n in needles)


def matches(pattern: str) -> Check:
    return lambda r, _: re.search(pattern, r.text, re.IGNORECASE) is not None


def number(n: int) -> Check:
    return matches(rf"\b{n}\b")


def called(*names: str) -> Check:
    return lambda r, _: all(n in {c.name for c in r.tool_calls} for n in names)


def no_tools() -> Check:
    return lambda r, _: not r.tool_calls


def no_remember() -> Check:
    return lambda r, _: "remember" not in {c.name for c in r.tool_calls}


def remembered(*needles: str) -> Check:
    """Every needle appears in some memory the agent saved during this case."""
    return lambda _, memories: all(
        any(n in text for source, text in memories if source == "agent") for n in needles
    )


# "Okay" is deliberately absent: it is a common loanword in Brazilian Portuguese.
EN_MARKERS = re.compile(
    r"\b(I|I've|I'll|you|your|saved|noted|let me know|anything else|else|note)\b", re.IGNORECASE
)


def portuguese() -> Check:
    """Reply did not drift into English (e.g. after English tool output)."""
    return lambda r, _: bool(r.text.strip()) and not EN_MARKERS.search(r.text)


# Speaking as the user: first-person verbs, or "meu/minha" + a fact about the user.
# ("minha memória", "meu banco de dados" and "seu gosto" are the assistant talking; fine.)
FIRST_PERSON = re.compile(
    r"\b(moro|torço|gosto de|tenho (?:dentista|consulta|alergia)|"
    r"(?:meu|minha) (?:time|esposa|filh[oa]|cachorro|aniversário|mãe|pai|cidade))\b",
    re.IGNORECASE,
)


def addresses_user() -> Check:
    """A confirmation talks to the user ("você mora..."), not as the user ("moro...")."""
    return lambda r, _: not FIRST_PERSON.search(r.text)


def forgotten(needle: str) -> Check:
    """No memory at all (seeded or saved) still contains the needle."""
    return lambda _, memories: not any(needle in text for _, text in memories)


def kept(needle: str) -> Check:
    """A still-true memory was not deleted."""
    return lambda _, memories: any(needle in text for _, text in memories)


def stored_once(needle: str) -> Check:
    return lambda _, memories: sum(needle in text for _, text in memories) == 1


# A realistic memory holds many unrelated facts; follow-up cases must find theirs among these.
DISTRACTORS = [
    "Meu time do coração é o Sport.",
    "Trabalho como engenheiro de software.",
    "Prefiro café sem açúcar.",
    "Minha mãe se chama Helena e mora em Recife.",
    "Meu pai gosta de pescar nos fins de semana.",
    "Tenho alergia a camarão.",
    "Minha cor favorita é azul.",
    "Uso VS Code como editor principal.",
    "Meu carro é um Onix prata.",
    "Corro 5 km toda segunda e quarta.",
    "Meu melhor amigo se chama Bruno.",
    "Gosto de ouvir jazz enquanto trabalho.",
    "Minha sogra adora orquídeas.",
    "Estou aprendendo japonês.",
    "Meu gato se chama Mingau.",
    "Viajei para Lisboa em 2024.",
    "Não gosto de filmes de terror.",
    "Meu livro favorito é Dom Casmurro.",
    "Minha sobrinha Laura faz balé.",
    "Minha tia Marta cozinha muito bem.",
    "Meu colega Felipe gosta de cerveja artesanal.",
    "Tenho consulta no oftalmologista todo ano em janeiro.",
    "Minha vizinha Clara tem dois cachorros.",
    "Meu primo Rafael mora no Canadá.",
    "Gosto de pizza de calabresa.",
]


@dataclass
class Case:
    group: str
    message: str
    checks: list[Check]
    seed: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)


CASES = [
    # single intent
    Case("single", "Oi, tudo bem?", [no_tools()]),
    Case("single", "Explique o que é uma API REST em uma frase.", [no_tools()]),
    Case("single", "Qual a capital da França?", [no_tools(), says("Paris")]),
    Case(
        "single",
        "Lembre que meu time favorito é o Sport.",
        [remembered("Sport"), portuguese(), addresses_user()],
    ),
    Case(
        "single",
        "Anota aí: tenho dentista na quinta às 15h.",
        [remembered("dentista"), portuguese(), addresses_user()],
    ),
    # Colloquial "Lembra que X" is an imperative ("remember that X"), not "do you remember?"
    Case(
        "single",
        "Lembra que eu moro em Recife.",
        [remembered("Recife"), portuguese(), addresses_user()],
    ),
    Case(
        "single",
        "Lembra que meu cachorro se chama Thor.",
        [remembered("Thor"), portuguese(), addresses_user()],
    ),
    Case("single", "Que dia é hoje?", [says(WEEKDAY_PT)]),
    Case("single", "Quantos dias faltam para o Natal?", [number(DAYS_TO_XMAS)]),
    Case(
        "single",
        "Como se chama o meu cachorro?",
        [no_tools(), says("Thor")],
        seed=["Meu cachorro se chama Thor."],
    ),
    # compound
    Case(
        "compound",
        "Que dia da semana é hoje e quantos dias faltam pro Natal?",
        [says(WEEKDAY_PT), number(DAYS_TO_XMAS)],
    ),
    Case(
        "compound",
        "Que editor eu uso e quantos dias faltam pro Natal?",
        [says("VS Code"), number(DAYS_TO_XMAS)],
        seed=["Eu uso VS Code no Windows."],
    ),
    Case(
        "compound",
        "Lembre que meu time é o Sport e me diga quantos dias faltam para o Ano Novo.",
        [remembered("Sport"), number(DAYS_TO_NEW_YEAR), portuguese(), addresses_user()],
    ),
    Case(
        "compound",
        "Qual a capital da França e que dia é hoje?",
        [says("Paris", WEEKDAY_PT), no_remember()],
    ),
    Case(
        "compound",
        "Guarde que minha esposa se chama Ana e que meu filho se chama Pedro.",
        [remembered("Ana", "Pedro"), portuguese(), addresses_user()],
    ),
    Case(
        "compound",
        "Me explica o que é Docker em uma frase e lembra que tenho dentista na quinta.",
        [remembered("dentista"), matches(r"cont[aêe]i?ner"), portuguese(), addresses_user()],
    ),
    Case(
        "compound",
        "Meu aniversário é 10 de março. Quantos dias faltam? E guarda essa data.",
        [
            number(days_to(3, 10)),
            remembered("10"),
        ],
    ),
    Case(
        "compound",
        "Que horas são agora e qual o nome do meu cachorro?",
        [says("Thor"), matches(r"\b\d{1,2}[:h]\d{2}\b")],
        seed=["Meu cachorro se chama Thor."],
    ),
    # session: requests in earlier turns were already handled and must not be repeated
    Case(
        "session",
        "Qual a capital da Itália?",
        [no_tools(), says("Roma")],
        history=[
            ("user", "Lembre que meu time favorito é o Sport."),
            ("assistant", "Pronto, anotei que seu time favorito é o Sport."),
        ],
    ),
    Case(
        "session",
        "Me dá uma dica rápida de estudo.",
        [no_tools(), portuguese()],
        history=[
            ("user", "Anota que tenho dentista na quinta às 15h."),
            ("assistant", "Anotado: dentista na quinta às 15h."),
        ],
    ),
    Case(
        "session",
        "E quantos dias faltam pro Natal?",
        [called("days_until"), no_remember(), number(DAYS_TO_XMAS), portuguese()],
        history=[
            ("user", "Guarda que meu aniversário é 10 de março."),
            ("assistant", "Guardei: seu aniversário é 10 de março."),
        ],
    ),
    # questions about stored facts are answered from memory, never re-saved
    Case(
        "question",
        "Onde eu moro?",
        [no_tools(), says("São Paulo")],
        seed=["Moro em São Paulo.", "Trabalho como engenheiro de software."],
    ),
    Case(
        "question",
        "Qual é o meu time do coração mesmo?",
        [no_tools(), says("Sport")],
        seed=["Meu time favorito é o Sport.", "Moro em Recife."],
    ),
    Case(
        "question",
        "Onde eu moro e quantos dias faltam pro Natal?",
        [no_remember(), called("days_until"), says("São Paulo"), number(DAYS_TO_XMAS)],
        seed=["Moro em São Paulo."],
    ),
    Case(
        "question",
        "Você lembra o nome da minha esposa?",
        [no_tools(), says("Ana")],
        seed=["Minha esposa se chama Ana."],
    ),
    # memory hygiene: no duplicates, contradictions replace old facts, explicit forgetting
    Case(
        "memory",
        "Lembre que meu time favorito é o Sport.",
        [stored_once("Sport"), portuguese()],
        seed=["Meu time favorito é o Sport."],
    ),
    Case(
        "memory",
        "Mudei de time, agora torço pro Náutico.",
        [remembered("Náutico"), forgotten("Sport"), portuguese(), addresses_user()],
        seed=["Meu time favorito é o Sport.", "Trabalho como engenheiro de software."],
    ),
    Case(
        "memory",
        "Me mudei pra São Paulo mês passado, anota aí.",
        [remembered("São Paulo"), forgotten("Recife"), portuguese(), addresses_user()],
        seed=["Moro em Recife.", "Minha mãe se chama Helena."],
    ),
    Case(
        "memory",
        "Me mudei pra São Paulo.",
        [remembered("São Paulo"), forgotten("Recife"), kept("Thor"), portuguese()],
        # Phrased differently from the new fact: similarity only ~0.58.
        seed=["Eu moro em Recife.", "Meu cachorro se chama Thor.", "Minha mãe se chama Helena."],
    ),
    Case(
        "memory",
        "Esquece aquilo da alergia a camarão, era engano.",
        [forgotten("camarão"), kept("Helena"), portuguese()],
        seed=["Tenho alergia a camarão.", "Minha mãe se chama Helena."],
    ),
    Case(
        "memory",
        "Guarda que a Ana faz aniversário em 12 de maio.",
        [remembered("maio"), kept("esposa se chama Ana"), portuguese(), addresses_user()],
        seed=["Minha esposa se chama Ana."],
    ),
    Case(
        "memory",
        "Anota que minha filha se chama Laura.",
        [remembered("Laura"), kept("Pedro"), portuguese(), addresses_user()],
        seed=["Meu filho se chama Pedro."],
    ),
    Case(
        "memory",
        "Também gosto de pizza de mussarela, guarda aí.",
        [remembered("mussarela"), kept("calabresa"), portuguese(), addresses_user()],
        seed=["Gosto de pizza de calabresa."],
    ),
    # follow-up: the fact is only findable with the previous exchange as context
    Case(
        "followup",
        "O que eu poderia levar de presente pra ela?",
        [says("chocolate")],
        seed=[
            *DISTRACTORS,
            "Minha irmã se chama Júlia.",
            "Júlia adora chocolate amargo.",
            "Minha esposa Ana gosta de vinho tinto.",
            "Meu chefe se chama Roberto.",
        ],
        history=[
            ("user", "Vou visitar minha irmã Júlia no sábado."),
            ("assistant", "Que ótimo! Aproveite a visita."),
        ],
    ),
    Case(
        "followup",
        "Quando é o aniversário dela?",
        [says("maio")],
        seed=[
            *DISTRACTORS,
            "Minha esposa se chama Ana.",
            "Ana faz aniversário em 12 de maio.",
            "Meu irmão Carlos faz aniversário em 3 de agosto.",
        ],
        history=[
            ("user", "Estou pensando numa surpresa pra minha esposa."),
            ("assistant", "Que legal! Posso ajudar com ideias."),
        ],
    ),
    Case(
        "followup",
        "Qual framework ele usa mesmo?",
        [says("FastAPI")],
        seed=[
            *DISTRACTORS,
            "Meu projeto atual se chama Origin.",
            "O Origin é construído com FastAPI e LangChain.",
            "Tenho reunião com o Roberto toda terça.",
        ],
        history=[
            ("user", "Preciso de ajuda com o Origin, meu projeto atual."),
            ("assistant", "Claro! No que posso ajudar?"),
        ],
    ),
]


async def run_case(case: Case, settings, embeddings) -> tuple[bool, float, ChatResult]:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    memory = VectorMemory(embeddings, client, f"eval-{uuid4().hex}")
    if case.seed:
        await memory.add(case.seed, {"source": "seed"})
    judge = make_chat_model(settings, temperature=0)
    tools = ToolRegistry.discover(ToolContext(settings, memory, llm=judge)).tools
    engine = LLMEngine.from_settings(settings, memory=memory, tools=tools)

    history = [ChatTurn(role=role, content=content) for role, content in case.history]
    start = time.perf_counter()
    result = await engine.generate(case.message, history)
    elapsed = time.perf_counter() - start

    stored = memory._store.get()
    # Archived (superseded) memories are out of recall, so they count as forgotten.
    memories = [
        (meta.get("source", ""), text)
        for meta, text in zip(stored["metadatas"], stored["documents"], strict=True)
        if not meta.get("archived")
    ]
    # Every case: the reply must not just parrot the user's message back.
    echoed = normalize(result.text).startswith(normalize(case.message))
    return not echoed and all(check(result, memories) for check in case.checks), elapsed, result


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning", choices=["true", "false", "none"])
    parser.add_argument("--routing", choices=["true", "false"])
    parser.add_argument("--routing-temperature", type=float)
    parser.add_argument("--contextual-recall", choices=["true", "false"])
    parser.add_argument("--only", choices=sorted({c.group for c in CASES}))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.model:
        overrides["ollama_model"] = args.model
    if args.temperature is not None:
        overrides["ollama_temperature"] = args.temperature
    if args.reasoning:
        overrides["ollama_reasoning"] = {"true": True, "false": False, "none": None}[args.reasoning]
    if args.routing:
        overrides["agent_tool_routing"] = args.routing == "true"
    if args.routing_temperature is not None:
        overrides["agent_routing_temperature"] = args.routing_temperature
    if args.contextual_recall:
        overrides["memory_contextual_recall"] = args.contextual_recall == "true"
    settings = get_settings().model_copy(update=overrides)
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )
    cases = [c for c in CASES if args.only in (None, c.group)]

    print(
        f"model={settings.ollama_model} temperature={settings.ollama_temperature} "
        f"reasoning={settings.ollama_reasoning} routing={settings.agent_tool_routing} "
        f"routing_temperature={settings.agent_routing_temperature} "
        f"contextual_recall={settings.memory_contextual_recall} runs={args.runs}\n"
    )
    await LLMEngine.from_settings(settings).generate("ok", use_memory=False, use_tools=False)

    totals: dict[str, list[int]] = {}
    latencies: list[float] = []
    for case in cases:
        passes = 0
        for _ in range(args.runs):
            try:
                ok, elapsed, result = await run_case(case, settings, embeddings)
            except Exception as exc:  # e.g. a transient Ollama runner crash: one failed run
                ok, elapsed = False, 0.0
                result = ChatResult(text=f"<{type(exc).__name__}: {str(exc)[:120]}>")
            passes += ok
            latencies.append(elapsed)
            if args.verbose or not ok:
                tools = [f"{c.name}({c.args})" for c in result.tool_calls]
                mark = "  ok " if ok else "  FAIL"
                print(f"{mark} {elapsed:4.1f}s tools={tools}\n        {result.text[:160]!r}")
        totals.setdefault(case.group, []).append(passes)
        print(f"{passes}/{args.runs}  [{case.group}] {case.message}")

    print()
    for group, results in totals.items():
        print(f"{group:9} {sum(results)}/{len(results) * args.runs}")
    latencies.sort()
    print(f"latency   median {latencies[len(latencies) // 2]:.1f}s, max {latencies[-1]:.1f}s")
    if datetime.now().astimezone().date() != TODAY:
        print("WARNING: the date changed during the run; date-based checks are unreliable.")


if __name__ == "__main__":
    asyncio.run(main())
