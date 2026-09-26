import os
import tempfile
from collections.abc import Iterator
from uuid import uuid4

# Keep the app's persistent Chroma store out of ./data while tests run.
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="origin-test-chroma-"))

import chromadb  # noqa: E402
import pytest  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from main import app  # noqa: E402
from origin.api.routes.chat import get_engine  # noqa: E402
from origin.api.routes.memory import get_memory  # noqa: E402
from origin.core import LLMEngine  # noqa: E402
from origin.memory import VectorMemory  # noqa: E402


@pytest.fixture
def memory() -> VectorMemory:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorMemory(DeterministicFakeEmbedding(size=16), client, f"test-{uuid4().hex}")


@pytest.fixture
def fake_engine(memory: VectorMemory) -> LLMEngine:
    model = FakeListChatModel(responses=["Olá, eu sou o Origin."])
    return LLMEngine(model, system_prompt="test", model_name="fake", memory=memory)


@pytest.fixture
def client(fake_engine: LLMEngine, memory: VectorMemory) -> Iterator[TestClient]:
    app.dependency_overrides[get_engine] = lambda: fake_engine
    app.dependency_overrides[get_memory] = lambda: memory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
