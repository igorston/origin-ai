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

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_temperature: float = 0.7

    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/.chroma"
    sqlite_path: str = "./data/origin.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
