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
from origin.core import ChatResult, LLMEngine  # noqa: E402
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


def remembered(*needles: str) -> Check:
    return lambda _, memories: all(any(n in m for m in memories) for n in needles)


@dataclass
class Case:
    group: str
    message: str
    checks: list[Check]
    seed: list[str] = field(default_factory=list)


CASES = [
    # single intent
    Case("single", "Oi, tudo bem?", [no_tools()]),
    Case("single", "Explique o que é uma API REST em uma frase.", [no_tools()]),
    Case("single", "Qual a capital da França?", [no_tools(), says("Paris")]),
    Case("single", "Lembre que meu time favorito é o Sport.", [remembered("Sport")]),
    Case("single", "Anota aí: tenho dentista na quinta às 15h.", [remembered("dentista")]),
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
        [remembered("Sport"), number(DAYS_TO_NEW_YEAR)],
    ),
    Case(
        "compound",
        "Qual a capital da França e que dia é hoje?",
        [says("Paris", WEEKDAY_PT)],
    ),
    Case(
        "compound",
        "Guarde que minha esposa se chama Ana e que meu filho se chama Pedro.",
        [remembered("Ana", "Pedro")],
    ),
    Case(
        "compound",
        "Me explica o que é Docker em uma frase e lembra que tenho dentista na quinta.",
        [remembered("dentista"), matches(r"cont[aêe]i?ner")],
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
]


async def run_case(case: Case, settings, embeddings) -> tuple[bool, float, ChatResult]:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    memory = VectorMemory(embeddings, client, f"eval-{uuid4().hex}")
    if case.seed:
        await memory.add(case.seed, {"source": "seed"})
    tools = ToolRegistry.discover(ToolContext(settings, memory)).tools
    engine = LLMEngine.from_settings(settings, memory=memory, tools=tools)

    start = time.perf_counter()
    result = await engine.generate(case.message)
    elapsed = time.perf_counter() - start

    stored = memory._store.get(where={"source": "agent"})["documents"] if memory.count() else []
    return all(check(result, stored) for check in case.checks), elapsed, result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning", choices=["true", "false", "none"])
    parser.add_argument("--routing", choices=["true", "false"])
    parser.add_argument("--only", choices=["single", "compound"])
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
    settings = get_settings().model_copy(update=overrides)
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )
    cases = [c for c in CASES if args.only in (None, c.group)]

    print(
        f"model={settings.ollama_model} temperature={settings.ollama_temperature} "
        f"reasoning={settings.ollama_reasoning} routing={settings.agent_tool_routing} "
        f"runs={args.runs}\n"
    )
    await LLMEngine.from_settings(settings).generate("ok", use_memory=False, use_tools=False)

    totals: dict[str, list[int]] = {}
    latencies: list[float] = []
    for case in cases:
        passes = 0
        for _ in range(args.runs):
            ok, elapsed, result = await run_case(case, settings, embeddings)
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


if __name__ == "__main__":
    asyncio.run(main())
