"""How large a context window this machine can run: the model's limit and the VRAM.

The KV cache grows linearly with the window: for qwen3:8b, 36 layers x 8 KV heads x
(128 + 128) dims x 2 bytes = 144 KiB per token. The chat and embedding models must
also fit in VRAM together, or Ollama swaps them on every call (+4-5 s each). So:

    max window ~= (VRAM total - other apps - chat model file - embedding file - overhead)
                  / KV bytes per token

`overhead` (CUDA contexts, compute buffers, Ollama's reserve) was calibrated on an
RTX 4070 Laptop (8 GB) with qwen3:8b + bge-m3: 6144 tokens kept both loaded, 7168 and
8192 made Ollama swap them. 1.15 GB puts the limit between the two (~6570 -> 6144).
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel

from origin.config import Settings

logger = logging.getLogger(__name__)

GIB = 1024**3
STEP = 1024  # windows are rounded down to a multiple of this
MIN_WINDOW = 2048
# Per loaded model, beyond its size_vram (measured: 299 MiB unaccounted for 2 runners).
RUNNER_BYTES = 150 * 1024**2
# Bytes per KV element by OLLAMA_KV_CACHE_TYPE (f16 is Ollama's default).
KV_TYPE_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

Source = Literal["config", "vram", "model", "fallback"]


class Capacity(BaseModel):
    window: int  # the num_ctx in use
    source: Source  # why: set by hand, limited by VRAM or by the model, or a fallback
    model_limit: int | None = None  # context length the model was trained for
    vram_limit: int | None = None  # ~largest window that fits in VRAM (with embeddings)
    vram_total_gb: float | None = None
    gpu: str | None = None
    kv_kib_per_token: float | None = None


@dataclass
class _Gpu:
    name: str
    total: int  # bytes
    used: int


def _gpus() -> list[_Gpu]:
    """NVIDIA GPUs via nvidia-smi (other vendors: unknown, so the fallback applies)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, used = (part.strip() for part in line.rsplit(",", 2))
        gpus.append(_Gpu(name, int(float(total)) * 1024**2, int(float(used)) * 1024**2))
    return gpus


def _kv_bytes_per_token(info: dict) -> float | None:
    arch = info.get("general.architecture", "")
    get = lambda key: info.get(f"{arch}.{key}")  # noqa: E731
    layers, kv_heads = get("block_count"), get("attention.head_count_kv")
    key_len, value_len = get("attention.key_length"), get("attention.value_length")
    if not key_len and get("embedding_length") and get("attention.head_count"):
        key_len = value_len = get("embedding_length") // get("attention.head_count")
    if not (layers and kv_heads and key_len):
        return None
    element = KV_TYPE_BYTES.get(os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16"), 2.0)
    return layers * kv_heads * (key_len + (value_len or key_len)) * element


def vram_window(
    free_bytes: float, chat_file: int, embed_file: int, overhead_gb: float, kv_per_token: float
) -> int:
    """Largest window (rounded down to STEP) whose KV cache fits next to both models."""
    room = free_bytes - chat_file - embed_file - overhead_gb * GIB
    return max(0, int(room / kv_per_token) // STEP * STEP)


def probe(settings: Settings) -> Capacity:
    """Ask Ollama and the GPU driver; never raises (the fallback window is used)."""
    configured = settings.ollama_num_ctx
    fallback = configured if isinstance(configured, int) else settings.ollama_num_ctx_fallback
    capacity = Capacity(window=fallback, source="config" if configured != "auto" else "fallback")
    try:
        with httpx.Client(base_url=settings.ollama_base_url, timeout=5) as client:
            show = client.post("/api/show", json={"model": settings.ollama_model})
            show.raise_for_status()
            info = show.json().get("model_info", {})
            tags = {m["name"]: m["size"] for m in client.get("/api/tags").json()["models"]}
            running = client.get("/api/ps").json().get("models", [])
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not read the model's limits from Ollama (%s)", exc)
        return capacity

    arch = info.get("general.architecture", "")
    capacity.model_limit = info.get(f"{arch}.context_length")
    kv = _kv_bytes_per_token(info)
    capacity.kv_kib_per_token = round(kv / 1024, 1) if kv else None

    def size(name: str) -> int:
        return tags.get(name) or tags.get(f"{name}:latest") or 0

    gpus = _gpus()
    chat_file, embed_file = size(settings.ollama_model), size(settings.ollama_embed_model)
    if gpus and kv and chat_file:
        total = sum(g.total for g in gpus)
        # VRAM taken by other programs; Ollama's own models are evicted when needed. Each
        # loaded model's runner also holds a CUDA context its size_vram leaves out.
        ollama_vram = sum(m.get("size_vram", 0) + RUNNER_BYTES for m in running)
        other = max(0, sum(g.used for g in gpus) - ollama_vram)
        capacity.vram_total_gb = round(total / GIB, 1)
        capacity.gpu = " + ".join(g.name for g in gpus)
        capacity.vram_limit = vram_window(
            total - other, chat_file, embed_file, settings.ollama_vram_overhead_gb, kv
        )

    if configured != "auto":
        return capacity
    model_limit = capacity.model_limit // STEP * STEP if capacity.model_limit else None
    if capacity.vram_limit is None:
        # Memory unknown (no NVIDIA GPU, or no model sizes): the model's full limit could be
        # far too much, so stay at the fallback, within the model's limit.
        capacity.window = min(fallback, model_limit or fallback)
        return capacity
    if model_limit and model_limit <= capacity.vram_limit:
        capacity.window, capacity.source = model_limit, "model"
    else:
        capacity.window, capacity.source = max(MIN_WINDOW, capacity.vram_limit), "vram"
    return capacity


_cache: dict[tuple, Capacity] = {}


def capacity_for(settings: Settings) -> Capacity:
    """Probed once per process and configuration (every chat client must agree)."""
    key = (
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_embed_model,
        settings.ollama_num_ctx,
        settings.ollama_num_ctx_fallback,
        settings.ollama_vram_overhead_gb,
    )
    if key not in _cache:
        _cache[key] = probe(settings)
        logger.info(
            "Context window: %d tokens (%s) | model limit %s | fits in VRAM ~%s | %s",
            _cache[key].window,
            _cache[key].source,
            _cache[key].model_limit,
            _cache[key].vram_limit,
            _cache[key].gpu or "no NVIDIA GPU detected",
        )
    return _cache[key]


def num_ctx(settings: Settings) -> int:
    """The window every chat client uses: the configured number, or the probed one."""
    configured = settings.ollama_num_ctx
    return configured if isinstance(configured, int) else capacity_for(settings).window
