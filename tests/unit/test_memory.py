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


async def test_add_skips_near_duplicates(memory: VectorMemory) -> None:
    first = await memory.add(["Meu time é o Sport."])
    again = await memory.add(["Meu time é o Sport."])
    assert again == first
    assert memory.count() == 1

    await memory.add(["Meu time é o Sport."], dedup=False)
    assert memory.count() == 2


async def test_archive_hides_from_recall_until_restored(memory: VectorMemory) -> None:
    (old_id,) = await memory.add(["Moro em Recife."])

    memory.archive(old_id, superseded_by="new-id")

    assert await memory.search("Moro em Recife.") == []
    (archived,) = await memory.search("Moro em Recife.", include_archived=True)
    assert archived.metadata["archived"] is True
    assert archived.metadata["superseded_by"] == "new-id"
    assert memory.count() == 1
    assert memory.count(include_archived=False) == 0
    assert await memory.add(["Moro em Recife."]) != [old_id]  # archived is not a duplicate

    assert memory.restore(old_id) is True
    assert old_id in [h.id for h in await memory.search("Moro em Recife.", k=2)]
    assert memory.restore("missing") is False


async def test_records_are_newest_first(memory: VectorMemory) -> None:
    await memory.add(["first"])
    await memory.add(["second"])
    assert [r.content for r in memory.records()] == ["second", "first"]
    assert [r.content for r in memory.records(limit=1, offset=1)] == ["first"]


def test_memory_api_list_restore_and_stats(client: TestClient, memory: VectorMemory) -> None:
    (memory_id,) = client.post("/memory", json={"texts": ["Moro em Recife."]}).json()["ids"]
    memory.archive(memory_id)

    assert client.get("/memory/stats").json() == {
        "collection": memory.collection,
        "count": 1,
        "active": 0,
        "archived": 1,
    }
    assert client.get("/memory").json()[0]["metadata"]["archived"] is True

    restored = client.post(f"/memory/{memory_id}/restore")
    assert restored.status_code == 200
    assert restored.json()["metadata"]["archived"] is False
    assert client.post("/memory/nope/restore").status_code == 404


async def test_update_rewrites_and_reembeds(memory: VectorMemory) -> None:
    (memory_id,) = await memory.add(["Moro em Recife."], {"source": "agent"})

    record = await memory.update(memory_id, "Moro em Olinda.")

    assert record.content == "Moro em Olinda."
    assert record.metadata["source"] == "agent"
    assert "edited_at" in record.metadata
    # Fake embeddings are hash-based: only the new text matches exactly now.
    (hit,) = await memory.search("Moro em Olinda.", k=1)
    assert hit.id == memory_id and hit.score == pytest.approx(1.0)
    assert await memory.update("missing", "x") is None


def test_memory_api_patch(client: TestClient, memory: VectorMemory) -> None:
    (memory_id,) = client.post("/memory", json={"texts": ["Moro em Recife."]}).json()["ids"]

    edited = client.patch(f"/memory/{memory_id}", json={"content": "  Moro em Olinda.  "})
    assert edited.status_code == 200
    assert edited.json()["content"] == "Moro em Olinda."

    archived = client.patch(f"/memory/{memory_id}", json={"archived": True}).json()
    assert archived["metadata"]["archived"] is True
    assert client.get("/memory/stats").json()["archived"] == 1

    both = client.patch(
        f"/memory/{memory_id}", json={"content": "Moro em Recife.", "archived": False}
    )
    assert both.json()["content"] == "Moro em Recife."
    assert both.json()["metadata"]["archived"] is False

    assert client.patch("/memory/nope", json={"content": "x"}).status_code == 404
    assert client.patch(f"/memory/{memory_id}", json={}).status_code == 422
    assert client.patch(f"/memory/{memory_id}", json={"content": ""}).status_code == 422


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
