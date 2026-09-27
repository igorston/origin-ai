# ruff: noqa: E501  (the case table reads better one case per line)
"""Eval for memory normalization ("Otimizar com IA"): faithful, atomic, same language.

Usage:
    python scripts/eval_curation.py            # 3 runs per case
    python scripts/eval_curation.py --runs 5 -v

A case passes when the rewrite keeps every required detail, splits into the expected
number of facts, and stays in the note's language. A fallback to the original note
(when the guard rejects the rewrite) is reported separately: safe, but not optimized.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from origin.config import get_settings  # noqa: E402
from origin.core import make_chat_model  # noqa: E402
from origin.memory.curator import MemoryCurator, resolve_relative_days  # noqa: E402


@dataclass
class Case:
    note: str
    must_keep: list[str]  # case-insensitive substrings that must survive
    facts: int  # expected number of facts
    must_not: tuple[str, ...] = ()  # e.g. translated words


CASES = [
    Case("meu cachorro thor e minha gata luna", ["thor", "luna"], 2),
    Case("Mudei de emprego, agora trabalho na Globant como dev sênior", ["globant", "dev", "nior"], 1),
    Case("tenho dentista quinta 15h e reunião com o Roberto sexta 10h", ["dentista", "15h", "roberto", "10h"], 2),
    Case("sou alergico a amendoim, camarao e dipirona", ["amendoim", "camar", "dipirona"], 1),
    Case("eu uso vscode no windows 11 e programo em python", ["code", "windows 11", "python"], 2),
    Case("tenho 2 filhos: pedro (8) e laura (5)", ["pedro", "laura", "8", "5"], 1),
    Case("i work at google as a data scientist and my wife is called emma", ["google", "data scientist", "emma"], 2, ("trabalho", "esposa")),
    Case("mi hermano carlos vive en madrid", ["carlos", "madrid"], 1, ("irmão", "mora")),
    Case("Meu time favorito é o Sport.", ["meu time favorito é o sport."], 1),
    Case("reunião com o time amanhã 14h", ["14h", "/"], 1, ("amanhã",)),
]  # fmt: skip


def evaluate(case: Case, facts: list[str]) -> tuple[bool, bool]:
    """(passed, fell_back)"""
    text = " ".join(facts).lower()
    fell_back = facts == [resolve_relative_days(case.note.strip())]
    ok = (
        all(k.lower() in text for k in case.must_keep)
        and len(facts) == case.facts
        and not any(w in text for w in case.must_not)
    )
    return ok, fell_back


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    curator = MemoryCurator(memory=None, llm=make_chat_model(get_settings(), temperature=0))  # type: ignore[arg-type]
    passed = fallbacks = total = 0
    latencies = []
    for case in CASES:
        results = []
        for _ in range(args.runs):
            start = time.perf_counter()
            facts = await curator.normalize(case.note)
            latencies.append(time.perf_counter() - start)
            ok, fell_back = evaluate(case, facts)
            results.append(ok)
            fallbacks += fell_back and not ok
            if args.verbose or not ok:
                print(f"  {'ok  ' if ok else 'FAIL'} {'(fallback) ' if fell_back else ''}{facts}")
        passed += sum(results)
        total += len(results)
        print(f"{sum(results)}/{len(results)}  {case.note}")
    latencies.sort()
    print(
        f"\ncuration  {passed}/{total}  (failures that safely fell back to the note: {fallbacks})"
    )
    print(f"latency   median {latencies[len(latencies) // 2]:.1f}s, max {latencies[-1]:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
