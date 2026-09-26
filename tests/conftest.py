from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from main import app
from origin.api.routes.chat import get_engine
from origin.core import LLMEngine


@pytest.fixture
def fake_engine() -> LLMEngine:
    model = FakeListChatModel(responses=["Olá, eu sou o Origin."])
    return LLMEngine(model, system_prompt="test", model_name="fake")


@pytest.fixture
def client(fake_engine: LLMEngine) -> Iterator[TestClient]:
    app.dependency_overrides[get_engine] = lambda: fake_engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
