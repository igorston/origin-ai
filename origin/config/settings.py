import re
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    origin_env: Literal["dev", "prod", "test"] = "dev"
    origin_host: str = "127.0.0.1"
    origin_port: int = 8000
    origin_log_level: str = "INFO"
    origin_locale: str = "en-US"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embed_model: str = "bge-m3"
    ollama_temperature: float = 0.7
    # None = model default; False disables "thinking" on reasoning models (e.g. qwen3).
    ollama_reasoning: bool | None = False
    # Seconds Ollama keeps models loaded after the last request (-1 = forever).
    ollama_keep_alive: int = 1800
    # Preload models on startup so the first request does not pay the load time.
    ollama_warmup: bool = True
    # Retries for transient Ollama failures (runner briefly unreachable, 503, dropped
    # connection). Total tries per request; the wait doubles from the backoff each time.
    ollama_retry_attempts: int = 3
    ollama_retry_backoff: float = 0.5

    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/.chroma"
    memory_collection: str = "origin"
    memory_top_k: int = 4
    memory_min_score: float = 0.45
    # Also search memory with the previous exchange, so follow-ups ("and hers?") find facts.
    memory_contextual_recall: bool = True
    # Calibrated on bge-m3: paraphrases score >= ~0.90. Contradictions phrased differently
    # score as low as ~0.58 ("Eu moro em Recife." / "Moro em São Paulo."), overlapping with
    # merely related facts, so the conflict threshold only preselects candidates; a yes/no
    # model check (0 false positives in 60 judgments) decides. Recalibrate for other models.
    memory_dedup_threshold: float = 0.92
    memory_conflict_threshold: float = 0.55

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

    @field_validator("ollama_base_url")
    @classmethod
    def _prefer_ipv4_loopback(cls, url: str) -> str:
        # Ollama listens on 127.0.0.1 only. "localhost" resolves to ::1 first, and on Windows
        # every new connection then waits ~2s for the IPv6 attempt to fail (measured: first
        # memory search 2.1s -> 75ms).
        return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
