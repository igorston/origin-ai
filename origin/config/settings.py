from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    origin_env: Literal["dev", "prod", "test"] = "dev"
    origin_host: str = "127.0.0.1"
    origin_port: int = 8000
    origin_log_level: str = "INFO"
    origin_locale: str = "en-US"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embed_model: str = "bge-m3"
    ollama_temperature: float = 0.7
    # None = model default; False disables "thinking" on reasoning models (e.g. qwen3).
    ollama_reasoning: bool | None = False

    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/.chroma"
    memory_collection: str = "origin"
    memory_top_k: int = 4
    memory_min_score: float = 0.45

    tools_enabled: bool = True
    tools_disabled: list[str] = []
    agent_max_tool_iterations: int = 5
    sqlite_path: str = "./data/origin.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
