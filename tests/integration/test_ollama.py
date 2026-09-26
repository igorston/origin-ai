"""Integration tests against a real local Ollama server (skipped if unavailable)."""

import httpx
import pytest

from origin.config import get_settings
from origin.core import LLMEngine

settings = get_settings()


def _model_available() -> bool:
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).json()
    except httpx.HTTPError:
        return False
    return any(m["name"].split(":")[0] == settings.ollama_model for m in tags.get("models", []))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _model_available(), reason="Ollama/model not available"),
]


async def test_generate_with_ollama() -> None:
    engine = LLMEngine.from_settings(settings)
    answer = await engine.generate("Say hello in one short sentence.")
    assert answer.strip()


async def test_stream_with_ollama() -> None:
    engine = LLMEngine.from_settings(settings)
    chunks = [chunk async for chunk in engine.stream("Count from 1 to 3.")]
    assert len(chunks) > 1
    assert "".join(chunks).strip()
