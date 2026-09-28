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
    # White-label brand file (names, logo, colors, persona); see brand/brand.example.json.
    origin_brand_path: str = "./brand/brand.json"
    # Authentication. "off": single user, meant for 127.0.0.1 (the default). "password":
    # accounts with separate conversations and memories, for a shared or exposed
    # server (put HTTPS in front). Users: `python -m origin.auth add-user NAME --admin`,
    # or ORIGIN_ADMIN_PASSWORD to create the first admin at startup.
    origin_auth: Literal["off", "password"] = "off"
    origin_admin_user: str = "admin"
    origin_admin_password: str | None = None
    # Signs the session cookies; generated and kept in the data folder when unset.
    origin_secret_key: str | None = None
    origin_session_days: int = 30
    # Send the cookie only over HTTPS (set it when serving behind TLS).
    origin_cookie_secure: bool = False

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embed_model: str = "bge-m3"
    ollama_temperature: float = 0.7
    # None = model default; False disables "thinking" on reasoning models (e.g. qwen3).
    ollama_reasoning: bool | None = False
    # Context window in tokens, or "auto": the model's limit, capped by what fits in VRAM
    # next to the embedding model (see origin/core/capacity.py; on an RTX 4070 8 GB with
    # qwen3:8b + bge-m3 that is 6144 — at 7168 and 8192 Ollama swapped the two models on
    # every call, +4-5 s each). Without a value Ollama used 4096 and silently cut the start
    # of long prompts. Every chat client uses the same value, or Ollama reloads the model.
    ollama_num_ctx: int | Literal["auto"] = "auto"
    # "auto" without an NVIDIA GPU or without Ollama at startup: Ollama's own default.
    ollama_num_ctx_fallback: int = 4096
    # VRAM kept free beyond both model files and the KV cache (CUDA contexts, compute
    # buffers, Ollama's reserve). Calibrated on the GPU above; raise it if models swap.
    ollama_vram_overhead_gb: float = 1.15
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
    # Internet access (web_search, fetch_url). Available here, but each request still has
    # to ask for it (use_web / "Acesso à internet" in the interface): nothing leaves the
    # machine unless the user turns it on. false removes the tools altogether.
    web_access: bool = True
    # duckduckgo: no key. searxng: a (self-hosted) instance at WEB_SEARCH_URL.
    # brave: the Brave Search API, key in WEB_SEARCH_API_KEY.
    web_search_provider: Literal["duckduckgo", "searxng", "brave"] = "duckduckgo"
    web_search_url: str | None = None
    web_search_api_key: str | None = None
    web_max_results: int = 5
    # Characters of a page given to the model (~2.5 per token: 4000 ~ 1600 tokens).
    web_fetch_max_chars: int = 4000
    web_timeout: float = 10.0
    # Let the agent read private/local addresses (intranets). Off: it cannot be made to
    # read the local Ollama, the router or cloud metadata endpoints.
    web_allow_private: bool = False
    web_user_agent: str = (
        "Mozilla/5.0 (compatible; OriginAssistant/0.1; +https://github.com/igorston/origin-ai)"
    )
    tools_disabled: list[str] = []
    agent_max_tool_iterations: int = 5
    # Extra "decide tools first" turn; fixes compound questions on small models (~+0.4s).
    agent_tool_routing: bool = True
    # Choosing tools is classification: sampling noise there only causes skipped tools.
    agent_routing_temperature: float = 0.0
    sqlite_path: str = "./data/origin.db"
    # Which embedding model indexed the memories, and thresholds calibrated for it.
    calibration_path: str = "./data/calibration.json"
    # JSON backups of every memory, written before a reindex.
    memory_backup_dir: str = "./data/backups"
    # Context optimization (see origin/core/context.py). Budgets are in tokens.
    context_window: int | None = None  # defaults to ollama_num_ctx (override for tests)
    context_reply_reserve: int = 1024  # kept free for the answer and tool rounds
    context_warn_at: float = 0.6  # UI warning, as a fraction of the usable budget
    context_compact_at: float = 0.75  # fold old messages into the summary past this...
    context_compact_target: float = 0.5  # ...until usage is back under this
    context_keep_recent: int = 6  # messages kept verbatim when folding
    context_min_recent: int = 2  # never fold below this
    context_summary_max_tokens: int = 800  # condensed past this (capped at 20% of budget)

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
