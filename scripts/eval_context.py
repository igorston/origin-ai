# ruff: noqa: E402, E501
"""Eval for context optimization: do facts survive repeated summarization?

Runs one long conversation through the real API (in-process) with a small context
window, so old messages are folded into the summary several times. Long-term memory
and tools are OFF, so the planted facts can only come back through the summary.

Usage:
    python scripts/eval_context.py                 # window 4096 tokens
    python scripts/eval_context.py --window 3000 -v
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PLANTED = [
    "Meu voo para Lisboa é no dia 14 de novembro, o código da reserva é XK7Q2P.",
    "O nome do meu gerente é Ricardo Almeida.",
    "Estou refatorando o billing_service.py; a função calcular_fatura tem um bug de arredondamento.",
    "Meu orçamento para o notebook novo é de R$ 7.500.",
]
FILLER = [
    "Explique em um parágrafo como funciona o protocolo HTTPS.",
    "Me dê três dicas para dormir melhor, com uma frase de explicação cada.",
    "O que é um índice em banco de dados e quando ele atrapalha?",
    "Resuma em um parágrafo a história da Revolução Industrial.",
    "Qual a diferença entre processos e threads?",
    "Explique o que é inflação para uma criança de 10 anos.",
    "Como funciona o algoritmo de ordenação quicksort?",
    "Quais são os benefícios de caminhar 30 minutos por dia?",
    "O que é o teorema de Pitágoras e um exemplo de uso?",
    "Explique o que é uma API REST e seus verbos principais.",
    "Por que o céu é azul? Responda em um parágrafo.",
    "Dê um exemplo de uso de list comprehension em Python e explique.",
    "O que é fotossíntese?",
    "Explique a diferença entre TCP e UDP.",
    "Quais cuidados tomar ao guardar senhas num banco de dados?",
    "O que é machine learning, em poucas frases?",
]
QUESTIONS = [
    ("Qual é o código da reserva do meu voo?", ["XK7Q2P"]),
    ("Para onde é o meu voo e em que dia?", ["Lisboa", "14"]),
    ("Como se chama o meu gerente?", ["Ricardo"]),
    (
        "Qual função do billing_service.py está com bug, e que bug é?",
        ["calcular_fatura", "arredond"],
    ),
    ("Qual é o meu orçamento para o notebook?", ["7.500", "7500"]),  # any of these
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="origin-eval-context-")
    os.environ.update(
        CONTEXT_WINDOW=str(args.window),
        SQLITE_PATH=f"{tmp}/origin.db",
        CHROMA_PERSIST_DIR=f"{tmp}/chroma",
        CALIBRATION_PATH=f"{tmp}/calibration.json",
        MEMORY_BACKUP_DIR=f"{tmp}/backups",
    )
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as client:
        estimate = app.state.context.base_tokens
        for _ in range(60):  # let warmup measure the fixed prompt
            if app.state.context.base_tokens != estimate:
                break
            time.sleep(0.5)
        print(
            f"window {args.window} tokens, fixed prompt {app.state.context.base_tokens} "
            f"(estimate was {estimate})\n"
        )
        session_id = client.post("/sessions").json()["id"]
        compactions = 0
        compressions: list = []
        peak = 0.0

        def say(message: str) -> dict:
            nonlocal compactions, peak
            response = client.post(
                "/chat",
                json={
                    "message": message,
                    "session_id": session_id,
                    "use_memory": False,
                    "use_tools": False,
                },
            )
            if response.status_code == 409:
                raise SystemExit(f"CLOSED early: {response.json()['detail']['reason']}")
            if response.status_code != 200:
                raise SystemExit(f"turn failed {response.status_code}: {response.text[:300]}")
            body = response.json()
            compactions += len(body["compactions"])
            compressions.extend(c for c in body["compactions"] if c["compressed"])
            ctx = body["context"]
            peak = max(peak, ctx["percent"])
            if args.verbose or body["compactions"]:
                mark = (
                    f"  [otimizado: {', '.join(str(c['folded']) for c in body['compactions'])} msgs]"
                    if body["compactions"]
                    else ""
                )
                print(
                    f"  {ctx['used']:5d}/{ctx['usable']} ({ctx['percent']:.0%}) {ctx['state']:8} {message[:50]!r}{mark}"
                )
            return body

        for fact in PLANTED:
            say(fact)
        for question in FILLER:
            say(question)

        print(
            f"\ncompactions: {compactions}, compressions: {len(compressions)}, peak usage: {peak:.0%}"
        )
        summary = client.get(f"/sessions/{session_id}").json()["summary"]
        print(f"summary (~{len(summary)} chars):\n  " + summary.replace("\n", "\n  "))

        hits = 0
        print()
        for question, expected in QUESTIONS:
            answer = say(question)["response"]
            ok = (
                any(e.lower() in answer.lower() for e in expected)
                if len(expected) == 2 and expected[1] == "7500"
                else all(e.lower() in answer.lower() for e in expected)
            )
            hits += ok
            print(f"  {'ok  ' if ok else 'FAIL'} {question}\n       -> {answer[:160]!r}")
        print(
            f"\ncontext  {hits}/{len(QUESTIONS)} planted facts recalled after {compactions} optimizations"
        )


if __name__ == "__main__":
    main()
