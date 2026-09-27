import json
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from main import app
from origin.api.routes.calibration import get_calibrator
from origin.memory import VectorMemory
from origin.memory.calibration import (
    CONTRADICTIONS,
    CalibrationStore,
    MemoryCalibrator,
    Thresholds,
)

DEFAULTS = Thresholds(dedup=0.92, conflict=0.55, min_score=0.45)


class StaticJudge:
    """Replaces exactly the labeled contradictions."""

    async def supersedes(self, old: str, new: str) -> bool:
        return (old, new) in CONTRADICTIONS


class BrokenEmbeddings(DeterministicFakeEmbedding):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("ollama down")


@pytest.fixture
def client_db() -> chromadb.ClientAPI:
    return chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))


def make(
    client_db, collection: str, tmp_path: Path, size: int = 16, model: str = "embed-a", **kwargs
) -> tuple[MemoryCalibrator, VectorMemory, list[Thresholds]]:
    embeddings = kwargs.pop("embeddings", DeterministicFakeEmbedding(size=size))
    memory = VectorMemory(embeddings, client_db, f"calibration-{collection}")
    applied: list[Thresholds] = []
    calibrator = MemoryCalibrator(
        memory,
        judge=kwargs.pop("judge", StaticJudge()),
        store=CalibrationStore(tmp_path / "calibration.json"),
        embed_model=model,
        chat_model="chat",
        defaults=DEFAULTS,
        apply=applied.append,
        backup_dir=tmp_path / "backups",
    )
    return calibrator, memory, applied


async def test_bootstrap_records_the_indexing_model(client_db, tmp_path: Path) -> None:
    calibrator, memory, _ = make(client_db, "c1", tmp_path)
    await memory.add(["Moro em Recife."])

    status = await calibrator.bootstrap()

    assert status.indexed_model == "embed-a"
    assert status.stored_dim == status.current_dim == 16
    assert status.needs_reindex is False
    assert status.calibrated is False and status.thresholds == DEFAULTS


async def test_model_switch_is_detected_and_fixed_by_reindex(client_db, tmp_path: Path) -> None:
    old, old_memory, _ = make(client_db, "c2", tmp_path, size=16, model="embed-a")
    ids = await old_memory.add(
        ["Moro em Recife.", "Meu cachorro se chama Thor."], {"source": "manual"}
    )
    old_memory.archive(ids[1])
    await old.bootstrap()

    # Restart with another embedding model (different dimension) over the same store.
    new, memory, _ = make(client_db, "c2", tmp_path, size=8, model="embed-b")
    status = await new.bootstrap()
    assert status.needs_reindex is True
    assert "16" in status.reason and "8" in status.reason

    result = await new.reindex()

    assert result.reindexed == 2
    backup = json.loads(Path(result.backup).read_text(encoding="utf-8"))
    assert {r["content"] for r in backup} == {"Moro em Recife.", "Meu cachorro se chama Thor."}
    status = await new.status()
    assert status.needs_reindex is False
    assert (status.indexed_model, status.stored_dim) == ("embed-b", 8)
    # Content, ids and metadata (archived flag included) survive the reindex.
    assert memory.get(ids[1]).metadata["archived"] is True
    assert memory.get(ids[0]).metadata["source"] == "manual"
    (hit,) = await memory.search("Moro em Recife.", k=1)
    assert hit.id == ids[0]


async def test_same_dimension_but_another_model_also_needs_reindex(
    client_db, tmp_path: Path
) -> None:
    old, memory, _ = make(client_db, "c3", tmp_path, model="embed-a")
    await memory.add(["x"])
    await old.bootstrap()

    new, _, _ = make(client_db, "c3", tmp_path, model="embed-c")  # same size, other model
    status = await new.status()
    assert status.needs_reindex is True
    assert "embed-a" in status.reason


async def test_failed_reindex_leaves_memories_untouched(client_db, tmp_path: Path) -> None:
    calibrator, memory, _ = make(client_db, "c4", tmp_path)
    await memory.add(["Moro em Recife."])
    broken, _, _ = make(client_db, "c4", tmp_path, embeddings=BrokenEmbeddings(size=16))

    with pytest.raises(ConnectionError):
        await broken.reindex()

    assert memory.count() == 1
    assert [r.content for r in memory.records()] == ["Moro em Recife."]


async def test_calibrate_apply_persist_and_reset(client_db, tmp_path: Path) -> None:
    calibrator, _, applied = make(client_db, "c5", tmp_path)

    report = await calibrator.calibrate(apply=False)
    assert applied == []  # suggestion only
    for value in report.suggested.model_dump().values():
        assert 0 <= value <= 1
    assert report.judge.replaced_contradictions == len(CONTRADICTIONS)
    assert report.judge.false_replacements == []
    assert set(report.scores) == {
        "paraphrase",
        "contradiction",
        "compatible",
        "unrelated",
        "relevant",
        "irrelevant",
    }

    await calibrator.calibrate(apply=True)
    assert applied[-1] == report.suggested
    assert (await calibrator.status()).calibrated is True

    # A restart re-applies the calibration for the same embedding model...
    again, _, reapplied = make(client_db, "c5", tmp_path)
    await again.bootstrap()
    assert reapplied == [report.suggested]
    # ...but not for another one.
    other, _, other_applied = make(client_db, "c5", tmp_path, model="embed-z")
    status = await other.bootstrap()
    assert other_applied == [] and status.calibrated is False

    assert calibrator.reset() == DEFAULTS
    assert applied[-1] == DEFAULTS


async def test_judge_false_replacements_are_warned(client_db, tmp_path: Path) -> None:
    class Trigger:
        async def supersedes(self, old: str, new: str) -> bool:
            return True  # archives everything, including facts that are still true

    calibrator, _, _ = make(client_db, "c6", tmp_path, judge=Trigger())
    report = await calibrator.calibrate()
    assert report.judge.false_replacements
    assert any("arquivaria" in w for w in report.warnings)


def test_calibration_api(client: TestClient, client_db, tmp_path: Path) -> None:
    calibrator, memory, applied = make(client_db, f"api-{uuid4().hex}", tmp_path)
    app.dependency_overrides[get_calibrator] = lambda: calibrator

    status = client.get("/memory/calibration").json()
    assert status["embed_model"] == "embed-a" and status["needs_reindex"] is False

    report = client.post("/memory/calibration/run", json={"apply": True}).json()
    assert client.get("/memory/calibration").json()["thresholds"] == report["suggested"]

    manual = {"dedup": 0.95, "conflict": 0.5, "min_score": 0.4}
    assert client.put("/memory/calibration/thresholds", json=manual).json() == manual
    assert applied[-1] == Thresholds(**manual)
    assert client.delete("/memory/calibration/thresholds").json() == DEFAULTS.model_dump()

    assert client.post("/memory/reindex").json()["reindexed"] == 0


def test_dimension_mismatch_is_a_409_with_guidance(client: TestClient, client_db) -> None:
    from origin.api.routes.memory import get_memory

    old = VectorMemory(DeterministicFakeEmbedding(size=16), client_db, "dims")
    old._store.add_texts(["Moro em Recife."])
    app.dependency_overrides[get_memory] = lambda: VectorMemory(
        DeterministicFakeEmbedding(size=8), client_db, "dims"
    )

    response = client.get("/memory/search", params={"q": "onde"})

    assert response.status_code == 409
    assert "reindexada" in response.json()["detail"]
