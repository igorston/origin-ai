import pytest
from fastapi.testclient import TestClient

from origin.memory import VectorMemory


async def test_add_search_delete(memory: VectorMemory) -> None:
    ids = await memory.add(["alpha", "beta"], {"source": "test"})
    assert len(ids) == 2
    assert memory.count() == 2

    hits = await memory.search("alpha", k=1)
    assert hits[0].content == "alpha"
    assert hits[0].id == ids[0]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata["source"] == "test"
    assert "created_at" in hits[0].metadata

    await memory.delete([ids[0]])
    assert memory.count() == 1


async def test_search_filters_by_min_score(memory: VectorMemory) -> None:
    await memory.add(["alpha", "beta"])
    hits = await memory.search("alpha", k=2, min_score=0.99)
    assert [h.content for h in hits] == ["alpha"]


def test_memory_api_roundtrip(client: TestClient) -> None:
    created = client.post(
        "/memory", json={"texts": ["Prefiro Python"], "metadata": {"tag": "pref"}}
    )
    assert created.status_code == 201
    (memory_id,) = created.json()["ids"]

    assert client.get("/memory/stats").json()["count"] == 1

    hits = client.get("/memory/search", params={"q": "Prefiro Python"}).json()
    assert hits[0]["id"] == memory_id
    assert hits[0]["metadata"]["tag"] == "pref"

    assert client.delete(f"/memory/{memory_id}").status_code == 204
    assert client.get("/memory/stats").json()["count"] == 0


def test_memory_api_validates_input(client: TestClient) -> None:
    assert client.post("/memory", json={"texts": []}).status_code == 422
    assert client.post("/memory", json={"texts": [""]}).status_code == 422
    assert client.get("/memory/search", params={"q": ""}).status_code == 422
