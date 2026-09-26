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
    # Seconds Ollama keeps models loaded after the last request (-1 = forever).
    ollama_keep_alive: int = 1800
    # Preload models on startup so the first request does not pay the load time.
    ollama_warmup: bool = True

    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/.chroma"
    memory_collection: str = "origin"
    memory_top_k: int = 4
    memory_min_score: float = 0.45
    # Also search memory with the previous exchange, so follow-ups ("and hers?") find facts.
    memory_contextual_recall: bool = True
    # Calibrated on bge-m3: paraphrases score >= ~0.90, contradictions ~0.74-0.87,
    # merely related facts ~0.62-0.70. Recalibrate if you change the embedding model.
    memory_dedup_threshold: float = 0.92
    memory_conflict_threshold: float = 0.72

    tools_enabled: bool = True
    tools_disabled: list[str] = []
    agent_max_tool_iterations: int = 5
    # Extra "decide tools first" turn; fixes compound questions on small models (~+0.4s).
    agent_tool_routing: bool = True
    # Choosing tools is classification: sampling noise there only causes skipped tools.
    agent_routing_temperature: float = 0.0
    sqlite_path: str = "./data/origin.db"
    # Most recent stored messages sent to the model as context for a session.
    session_history_limit: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
