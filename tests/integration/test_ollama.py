"""Integration tests against a real local Ollama server (skipped if unavailable)."""

from uuid import uuid4

import chromadb
import httpx
import pytest
from chromadb.config import Settings as ChromaSettings
from langchain_ollama import OllamaEmbeddings

from origin.config import get_settings
from origin.core import LLMEngine
from origin.memory import VectorMemory

settings = get_settings()


def _installed_models() -> set[str]:
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).json()
    except httpx.HTTPError:
        return set()
    return {m["name"].split(":")[0] for m in tags.get("models", [])}


MODELS = _installed_models()
requires_llm = pytest.mark.skipif(settings.ollama_model not in MODELS, reason="LLM not available")
requires_embeddings = pytest.mark.skipif(
    settings.ollama_embed_model not in MODELS, reason="Embedding model not available"
)
pytestmark = pytest.mark.integration


@pytest.fixture
def real_memory() -> VectorMemory:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )
    return VectorMemory(embeddings, client, f"it-{uuid4().hex}")


@requires_llm
async def test_generate_with_ollama() -> None:
    engine = LLMEngine.from_settings(settings)
    answer = await engine.generate("Say hello in one short sentence.")
    assert answer.strip()


@requires_llm
async def test_stream_with_ollama() -> None:
    engine = LLMEngine.from_settings(settings)
    chunks = [chunk async for chunk in engine.stream("Count from 1 to 3.")]
    assert len(chunks) > 1
    assert "".join(chunks).strip()


@requires_embeddings
async def test_semantic_recall_in_portuguese(real_memory: VectorMemory) -> None:
    await real_memory.add(
        [
            "Meu nome é Douglas e moro no Brasil.",
            "Prefiro Python com FastAPI para backends.",
            "Meu cachorro se chama Thor.",
            "A reunião de sprint é toda segunda às 10h.",
        ]
    )
    cases = {
        "Qual é o nome do meu pet?": "Thor",
        "Qual linguagem eu gosto de usar?": "Python",
        "Quando é a reunião de sprint?": "segunda",
    }
    for query, expected in cases.items():
        hits = await real_memory.search(query, k=1, min_score=settings.memory_min_score)
        assert hits, f"no memory above threshold for {query!r}"
        assert expected in hits[0].content

    assert not await real_memory.search(
        "Qual a capital da França?", k=4, min_score=settings.memory_min_score
    )


@requires_llm
@requires_embeddings
async def test_chat_uses_memory(real_memory: VectorMemory) -> None:
    await real_memory.add(["Meu cachorro se chama Thor."])
    engine = LLMEngine.from_settings(settings, memory=real_memory)
    answer = await engine.generate("Como se chama o meu cachorro?")
    assert "thor" in answer.lower()
