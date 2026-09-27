import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

# Keep the app's persistent stores out of ./data and skip model warmup while tests run.
_tmp = tempfile.mkdtemp(prefix="origin-test-")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(_tmp, "chroma"))
os.environ.setdefault("SQLITE_PATH", os.path.join(_tmp, "origin.db"))
os.environ.setdefault("CALIBRATION_PATH", os.path.join(_tmp, "calibration.json"))
os.environ.setdefault("MEMORY_BACKUP_DIR", os.path.join(_tmp, "backups"))
os.environ.setdefault("OLLAMA_WARMUP", "false")

import chromadb  # noqa: E402
import pytest  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from main import app  # noqa: E402
from origin.api.routes.chat import get_engine, get_sessions  # noqa: E402
from origin.api.routes.memory import get_curator, get_memory  # noqa: E402
from origin.core import LLMEngine  # noqa: E402
from origin.memory import VectorMemory  # noqa: E402
from origin.memory.curator import MemoryCurator  # noqa: E402
from origin.memory.storage import SessionStore  # noqa: E402


@pytest.fixture
def memory() -> VectorMemory:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorMemory(DeterministicFakeEmbedding(size=16), client, f"test-{uuid4().hex}")


@pytest.fixture
def fake_engine(memory: VectorMemory) -> LLMEngine:
    model = FakeListChatModel(responses=["Olá, eu sou o Origin."])
    return LLMEngine(model, system_prompt="test", model_name="fake", memory=memory)


@pytest.fixture
def sessions(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


@pytest.fixture
def curator(memory: VectorMemory) -> MemoryCurator:
    """No model by default: optimize=True stores texts as written. Tests that exercise
    normalization build a curator with a scripted model."""
    return MemoryCurator(memory, llm=None)


@pytest.fixture
def client(
    fake_engine: LLMEngine, memory: VectorMemory, sessions: SessionStore, curator: MemoryCurator
) -> Iterator[TestClient]:
    app.dependency_overrides[get_engine] = lambda: fake_engine
    app.dependency_overrides[get_memory] = lambda: memory
    app.dependency_overrides[get_sessions] = lambda: sessions
    app.dependency_overrides[get_curator] = lambda: curator
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
