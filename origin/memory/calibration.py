"""Memory calibration for the current models.

The memory thresholds depend on the embedding model's score distribution, and the
"does the new fact replace the old one?" check depends on the chat model. When either
model changes, stored vectors may become incompatible (another dimension or space) and
the thresholds stop meaning what they meant. This module:

- tracks which embedding model indexed the memories (data/calibration.json),
- measures the current embedding model on a built-in labeled set and derives the
  dedup / conflict / recall thresholds,
- checks the conflict judge of the current chat model on labeled pairs.
"""

import json
import logging
import math
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from origin.branding import translator
from origin.i18n import Translator
from origin.memory.vectorstore import VectorMemory

logger = logging.getLogger(__name__)


class SupersedeJudge(Protocol):
    """The conflict check of the current chat model (MemoryCurator implements it)."""

    async def supersedes(self, old: str, new: str) -> bool: ...


# ---------------------------------------------------------------- labeled set
# Same fact, different wording: should be merged (dedup).
PARAPHRASES = [
    ("Meu time favorito é o Sport.", "O meu time do coração é o Sport."),
    ("Tenho dentista na quinta às 15h.", "Tenho consulta no dentista quinta-feira às 15 horas."),
    ("Minha esposa se chama Ana.", "O nome da minha esposa é Ana."),
    ("Minha irmã Júlia adora chocolate amargo.", "Minha irmã Júlia adora chocolate amargo"),
    ("Moro em Recife.", "Eu moro em Recife."),
    ("Trabalho na Globant como dev sênior.", "Sou dev sênior na Globant."),
    ("My dog is called Rex.", "My dog's name is Rex."),
    ("I live in Lisbon.", "I'm living in Lisbon."),
]
# Same attribute, new value: must reach the judge (and be replaced), never be merged.
CONTRADICTIONS = [
    ("Meu time favorito é o Sport.", "Meu time favorito é o Náutico."),
    ("Moro em Recife.", "Moro em São Paulo."),
    ("Eu moro em Recife.", "Moro em São Paulo."),
    ("Trabalho na Stellar Gaming.", "Agora trabalho na Globant."),
    ("Tenho 34 anos.", "Tenho 35 anos."),
    ("Meu editor favorito é o Vim.", "Meu editor favorito é o VS Code."),
    ("I live in Lisbon.", "I live in Porto."),
    ("My favorite band is Queen.", "My favorite band is Coldplay."),
]
# Related but both true: the judge must keep them.
COMPATIBLE = [
    ("Minha esposa se chama Ana.", "Ana faz aniversário em 12 de maio."),
    ("Meu cachorro se chama Thor.", "Meu gato se chama Mingau."),
    ("Meu filho se chama Pedro.", "Minha filha se chama Laura."),
    ("Gosto de pizza de calabresa.", "Gosto de pizza de mussarela."),
    ("Moro em Recife.", "Trabalho em Recife."),
    ("Tenho dentista na quinta às 15h.", "Tenho reunião na quinta às 15h."),
    ("Minha irmã se chama Júlia.", "Júlia adora chocolate amargo."),
    ("I have a brother called Tom.", "I have a sister called Mary."),
]
UNRELATED = [
    ("Meu time favorito é o Sport.", "Tenho alergia a camarão."),
    ("Moro em Recife.", "Meu cachorro se chama Thor."),
    ("Uso VS Code como editor.", "Minha mãe se chama Helena."),
    ("I live in Lisbon.", "My wife is called Emma."),
]
# Recall: each query must find its fact among all the others.
MEMORY_POOL = [
    "Moro em São Paulo.",
    "Meu cachorro se chama Thor.",
    "Programo em Python.",
    "A reunião de sprint é toda segunda às 10h.",
    "Tenho alergia a camarão.",
    "Meu time favorito é o Sport.",
    "Minha esposa se chama Ana.",
    "Uso VS Code como editor.",
    "Minha mãe se chama Helena e mora em Recife.",
    "Corro 5 km toda segunda e quarta.",
    "Meu livro favorito é Dom Casmurro.",
    "Gosto de ouvir jazz enquanto trabalho.",
    "Meu carro é um Onix prata.",
    "Estou aprendendo japonês.",
    "My sister Julia lives in Porto.",
    "I work at Google as a data scientist.",
]
RELEVANT_QUERIES = [
    ("Onde eu moro?", "Moro em São Paulo."),
    ("Qual o nome do meu pet?", "Meu cachorro se chama Thor."),
    ("Que linguagem de programação eu uso?", "Programo em Python."),
    ("Quando é a reunião de sprint?", "A reunião de sprint é toda segunda às 10h."),
    ("Que comida eu devo evitar?", "Tenho alergia a camarão."),
    ("Pra que time eu torço?", "Meu time favorito é o Sport."),
    ("Como se chama minha esposa?", "Minha esposa se chama Ana."),
    ("Que editor eu uso?", "Uso VS Code como editor."),
    ("Where does my sister live?", "My sister Julia lives in Porto."),
    ("What is my job?", "I work at Google as a data scientist."),
]
IRRELEVANT_QUERIES = [
    "Qual a capital da França?",
    "Explique o que é Docker em uma frase.",
    "Me dá uma dica rápida de estudo.",
    "Quanto é 17 vezes 23?",
    "What is the tallest mountain in the world?",
]

DEDUP_MARGIN = 0.02
RECALL_MARGIN = 0.05
CONFLICT_MARGIN = 0.03

# ---------------------------------------------------------------- models


class Thresholds(BaseModel):
    dedup: float
    conflict: float
    min_score: float


class ScoreStats(BaseModel):
    min: float
    mean: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "ScoreStats":
        return cls(
            min=round(min(values), 3),
            mean=round(statistics.fmean(values), 3),
            max=round(max(values), 3),
        )


class JudgeCheck(BaseModel):
    replaced_contradictions: int  # true positives
    contradictions: int
    false_replacements: list[str]  # compatible facts the judge would wrongly archive
    compatible: int


class CalibrationReport(BaseModel):
    embed_model: str
    chat_model: str
    measured_at: str
    duration_s: float
    current: Thresholds
    suggested: Thresholds
    scores: dict[
        str, ScoreStats
    ]  # paraphrase, contradiction, compatible, unrelated, relevant, irrelevant
    recall_hits: int  # relevant queries whose fact scores >= suggested min_score
    recall_total: int
    judge: JudgeCheck | None = None
    warnings: list[str] = []


class CalibrationState(BaseModel):
    """What is persisted in data/calibration.json."""

    embed_model: str | None = None  # model that produced the stored vectors
    embed_dim: int | None = None
    indexed_at: str | None = None
    thresholds: Thresholds | None = None  # applied calibration...
    calibrated_for: str | None = None  # ...valid only for this embedding model
    calibrated_at: str | None = None
    report: CalibrationReport | None = None


class IndexStatus(BaseModel):
    embed_model: str  # configured now
    chat_model: str
    indexed_model: str | None  # that produced the stored vectors
    stored_dim: int | None
    current_dim: int | None  # None when Ollama is unreachable
    count: int
    needs_reindex: bool
    reason: str | None = None
    calibrated: bool  # the active thresholds were measured for the current embedding model
    calibrated_at: str | None = None
    thresholds: Thresholds  # active now
    defaults: Thresholds  # from settings
    report: CalibrationReport | None = None


class ReindexResult(BaseModel):
    reindexed: int
    embed_model: str
    backup: str | None
    duration_s: float


class CalibrationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> CalibrationState:
        if not self.path.exists():
            return CalibrationState()
        try:
            return CalibrationState.model_validate_json(self.path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("Ignoring unreadable calibration file %s", self.path, exc_info=True)
            return CalibrationState()

    def save(self, state: CalibrationState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- measuring


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


async def _pair_scores(embeddings: Embeddings, pairs: Sequence[tuple[str, str]]) -> list[float]:
    vectors = await embeddings.aembed_documents([text for pair in pairs for text in pair])
    return [_cosine(vectors[2 * i], vectors[2 * i + 1]) for i in range(len(pairs))]


def _between(low: float, high: float, fallback: float) -> float:
    """Midpoint of a gap, or `fallback` when the two classes overlap."""
    return (low + high) / 2 if high > low else fallback


async def measure(
    embeddings: Embeddings,
    judge: SupersedeJudge | None,
    current: Thresholds,
    embed_model: str,
    chat_model: str,
    t: Translator | None = None,
) -> CalibrationReport:
    t = t or translator()  # the warnings are shown in the interface
    started = time.perf_counter()
    paraphrase = await _pair_scores(embeddings, PARAPHRASES)
    contradiction = await _pair_scores(embeddings, CONTRADICTIONS)
    compatible = await _pair_scores(embeddings, COMPATIBLE)
    unrelated = await _pair_scores(embeddings, UNRELATED)

    pool = await embeddings.aembed_documents(MEMORY_POOL)
    by_fact = dict(zip(MEMORY_POOL, pool, strict=True))
    relevant = []
    for query, fact in RELEVANT_QUERIES:
        q = await embeddings.aembed_query(query)
        relevant.append(_cosine(q, by_fact[fact]))
    irrelevant = []
    for query in IRRELEVANT_QUERIES:
        q = await embeddings.aembed_query(query)
        irrelevant.append(max(_cosine(q, v) for v in pool))

    warnings: list[str] = []
    not_duplicates = contradiction + compatible + unrelated

    # Dedup: a false merge loses a fact, a missed merge only leaves a duplicate, so keep a
    # clear margin above the most similar pair of *different* facts, even if that leaves
    # the closest paraphrases unmerged (bge-m3: different facts up to 0.897, paraphrases
    # from 0.904 — a midpoint would sit 0.004 from a false merge).
    dedup = max(
        max(not_duplicates) + DEDUP_MARGIN, _between(max(not_duplicates), min(paraphrase), 0)
    )
    if min(paraphrase) <= max(not_duplicates):
        warnings.append(t("calibration.dedup_overlap"))

    # Conflict: only preselects candidates for the judge, so it must sit below every
    # contradiction; extra candidates just cost one cheap judge call each.
    conflict = max(0.0, min(contradiction) - CONFLICT_MARGIN)

    # Recall: an irrelevant memory in the prompt is noise, a missed one is a wrong answer.
    # These are single, direct questions; compound and follow-up messages score lower
    # (e.g. 0.44 vs 0.56 for the same fact), so stay a margin below the weakest one.
    min_score = min(relevant) - RECALL_MARGIN
    if min(relevant) <= max(irrelevant):
        warnings.append(t("calibration.recall_overlap"))
    suggested = Thresholds(
        dedup=round(min(dedup, 0.99), 3),
        conflict=round(conflict, 3),
        min_score=round(max(min_score, 0.0), 3),
    )

    judge_check = None
    if judge is not None:
        replaced = [await judge.supersedes(old, new) for old, new in CONTRADICTIONS]
        wrongly = [f"{old} → {new}" for old, new in COMPATIBLE if await judge.supersedes(old, new)]
        judge_check = JudgeCheck(
            replaced_contradictions=sum(replaced),
            contradictions=len(CONTRADICTIONS),
            false_replacements=wrongly,
            compatible=len(COMPATIBLE),
        )
        if wrongly:
            warnings.append(t("calibration.false_replacements", count=len(wrongly)))
        if sum(replaced) < len(CONTRADICTIONS) / 2:
            warnings.append(t("calibration.weak_judge"))

    return CalibrationReport(
        embed_model=embed_model,
        chat_model=chat_model,
        measured_at=now_iso(),
        duration_s=round(time.perf_counter() - started, 1),
        current=current,
        suggested=suggested,
        scores={
            "paraphrase": ScoreStats.of(paraphrase),
            "contradiction": ScoreStats.of(contradiction),
            "compatible": ScoreStats.of(compatible),
            "unrelated": ScoreStats.of(unrelated),
            "relevant": ScoreStats.of(relevant),
            "irrelevant": ScoreStats.of(irrelevant),
        },
        recall_hits=sum(score >= suggested.min_score for score in relevant),
        recall_total=len(relevant),
        judge=judge_check,
        warnings=warnings,
    )


# ---------------------------------------------------------------- orchestration


class MemoryCalibrator:
    """Keeps the memory index and thresholds consistent with the configured models."""

    def __init__(
        self,
        memory: "VectorMemory",
        judge: SupersedeJudge | None,
        store: CalibrationStore,
        embed_model: str,
        chat_model: str,
        defaults: Thresholds,
        apply: Callable[[Thresholds], None],
        backup_dir: str | Path,
    ) -> None:
        self.memory = memory
        self.judge = judge
        self.store = store
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.defaults = defaults
        self._apply = apply
        self.active = defaults
        self.backup_dir = Path(backup_dir)

    def _set_active(self, thresholds: Thresholds) -> None:
        self.active = thresholds
        self._apply(thresholds)

    async def _current_dim(self) -> int | None:
        try:
            return await self.memory.current_dim()
        except Exception:
            logger.warning("Could not probe the embedding model", exc_info=True)
            return None

    async def bootstrap(self, probe: bool = True) -> IndexStatus:
        """On startup: apply saved thresholds when they were calibrated for the current
        embedding model, and record which model indexed a legacy/empty store. `probe`
        (query the embedding model for its dimension) is skipped when Ollama is down."""
        state = self.store.load()
        if state.thresholds and state.calibrated_for == self.embed_model:
            self._set_active(state.thresholds)
        if state.embed_model is None:
            stored = self.memory.stored_dim()
            current = await self._current_dim() if probe and stored is not None else None
            # Unknown history: an empty store, or vectors of the current dimension, are
            # taken as indexed by the current model.
            if stored is None or (current is not None and stored == current):
                state.embed_model, state.embed_dim = self.embed_model, current or stored
                state.indexed_at = now_iso()
                self.store.save(state)
        status = await self.status(probe=probe)
        if status.needs_reindex:
            logger.warning("Memory needs reindexing: %s", status.reason)
        elif not status.calibrated and status.count:
            logger.info("Memory thresholds are defaults, not calibrated for %s", self.embed_model)
        return status

    async def status(self, probe: bool = True) -> IndexStatus:
        state = self.store.load()
        count = self.memory.count()
        stored = self.memory.stored_dim() if count else None
        current = await self._current_dim() if probe and count else None
        reason = None
        if count and stored and current and stored != current:
            reason = translator()(
                "calibration.dim_mismatch", stored=stored, model=self.embed_model, current=current
            )
        elif count and state.embed_model and state.embed_model != self.embed_model:
            reason = translator()(
                "calibration.model_changed", indexed=state.embed_model, current=self.embed_model
            )
        return IndexStatus(
            embed_model=self.embed_model,
            chat_model=self.chat_model,
            indexed_model=state.embed_model,
            stored_dim=stored,
            current_dim=current,
            count=count,
            needs_reindex=reason is not None,
            reason=reason,
            calibrated=state.thresholds is not None and state.calibrated_for == self.embed_model,
            calibrated_at=state.calibrated_at if state.calibrated_for == self.embed_model else None,
            thresholds=self.active,
            defaults=self.defaults,
            report=state.report,
        )

    def backup(self) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        path = self.backup_dir / f"memory-{datetime.now():%Y%m%d-%H%M%S}.json"
        records = [r.model_dump() for r in self.memory.export()]
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    async def reindex(self) -> ReindexResult:
        started = time.perf_counter()
        backup = self.backup() if self.memory.count() else None
        count = await self.memory.reindex()
        state = self.store.load()
        state.embed_model = self.embed_model
        state.embed_dim = await self._current_dim()
        state.indexed_at = now_iso()
        if state.calibrated_for != self.embed_model:
            # Thresholds measured for another model no longer mean anything.
            state.thresholds = state.calibrated_for = state.calibrated_at = None
            self._set_active(self.defaults)
        self.store.save(state)
        logger.info("Reindexed %d memories with %s (backup: %s)", count, self.embed_model, backup)
        return ReindexResult(
            reindexed=count,
            embed_model=self.embed_model,
            backup=str(backup) if backup else None,
            duration_s=round(time.perf_counter() - started, 1),
        )

    async def calibrate(self, apply: bool = False) -> CalibrationReport:
        report = await measure(
            self.memory.embeddings, self.judge, self.active, self.embed_model, self.chat_model
        )
        state = self.store.load()
        state.report = report
        if apply:
            state.thresholds, state.calibrated_for = report.suggested, self.embed_model
            state.calibrated_at = report.measured_at
            self._set_active(report.suggested)
        self.store.save(state)
        return report

    def apply(self, thresholds: Thresholds) -> Thresholds:
        """Apply thresholds chosen by hand (e.g. the suggested ones, tweaked)."""
        state = self.store.load()
        state.thresholds, state.calibrated_for, state.calibrated_at = (
            thresholds,
            self.embed_model,
            now_iso(),
        )
        self.store.save(state)
        self._set_active(thresholds)
        return thresholds

    def reset(self) -> Thresholds:
        """Go back to the thresholds from the settings (.env)."""
        state = self.store.load()
        state.thresholds = state.calibrated_for = state.calibrated_at = None
        self.store.save(state)
        self._set_active(self.defaults)
        return self.defaults
