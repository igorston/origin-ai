import httpx
import pytest

from origin.config import Settings
from origin.core import capacity
from origin.core.capacity import GIB, _Gpu, _kv_bytes_per_token, probe, vram_window

QWEN3_8B = {
    "general.architecture": "qwen3",
    "qwen3.context_length": 40960,
    "qwen3.block_count": 36,
    "qwen3.attention.head_count": 32,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
}
QWEN_FILE, BGE_FILE = 5_225_388_164, 1_157_672_605  # bytes, from /api/tags
RTX_4070_8GB = 8188 * 1024**2


def test_kv_cache_per_token() -> None:
    assert _kv_bytes_per_token(QWEN3_8B) == 144 * 1024  # 36 x 8 x (128 + 128) x 2 bytes
    no_key_length = {k: v for k, v in QWEN3_8B.items() if "_length" not in k or "context" in k}
    no_key_length["qwen3.embedding_length"] = 4096  # head dim = 4096 / 32 = 128
    assert _kv_bytes_per_token(no_key_length) == 144 * 1024
    assert _kv_bytes_per_token({"general.architecture": "x"}) is None


def test_vram_window_matches_what_was_measured_on_an_8gb_gpu() -> None:
    # Measured: 6144 kept qwen3:8b and bge-m3 loaded together; 7168 made Ollama swap them.
    window = vram_window(RTX_4070_8GB, QWEN_FILE, BGE_FILE, 1.15, 144 * 1024)
    assert window == 6144
    # Another program holding 1 GB leaves room for less.
    assert vram_window(RTX_4070_8GB - GIB, QWEN_FILE, BGE_FILE, 1.15, 144 * 1024) < 6144
    assert vram_window(4 * GIB, QWEN_FILE, BGE_FILE, 1.15, 144 * 1024) == 0


def fake_ollama(monkeypatch, running=(), show_status=200) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(show_status, json={"model_info": QWEN3_8B})
        if request.url.path == "/api/tags":
            models = [
                {"name": "qwen3:8b", "size": QWEN_FILE},
                {"name": "bge-m3:latest", "size": BGE_FILE},
            ]
            return httpx.Response(200, json={"models": models})
        return httpx.Response(200, json={"models": list(running)})

    real = httpx.Client
    monkeypatch.setattr(
        capacity.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def test_auto_window_is_limited_by_vram(monkeypatch) -> None:
    fake_ollama(monkeypatch)
    monkeypatch.setattr(capacity, "_gpus", lambda: [_Gpu("RTX 4070", RTX_4070_8GB, 0)])
    result = probe(Settings(ollama_num_ctx="auto"))
    assert (result.window, result.source) == (6144, "vram")
    assert result.model_limit == 40960 and result.vram_total_gb == 8.0


def test_loaded_models_do_not_count_as_other_programs(monkeypatch) -> None:
    # Both models loaded (as measured: 6542 MiB used): the same window as an empty GPU.
    running = [{"size_vram": 5610 * 1024**2}, {"size_vram": 633 * 1024**2}]
    fake_ollama(monkeypatch, running)
    monkeypatch.setattr(capacity, "_gpus", lambda: [_Gpu("RTX", RTX_4070_8GB, 6542 * 1024**2)])
    assert probe(Settings(ollama_num_ctx="auto")).window == 6144


def test_a_big_gpu_is_limited_by_the_model(monkeypatch) -> None:
    fake_ollama(monkeypatch)
    monkeypatch.setattr(capacity, "_gpus", lambda: [_Gpu("A100", 80 * GIB, 0)])
    result = probe(Settings(ollama_num_ctx="auto"))
    assert (result.window, result.source) == (40960, "model")


@pytest.mark.parametrize("ollama_up", [True, False])
def test_unknown_memory_falls_back_to_a_safe_window(monkeypatch, ollama_up: bool) -> None:
    # No NVIDIA GPU (or Ollama unreachable): not the model's 40960, which may not fit.
    fake_ollama(monkeypatch, show_status=200 if ollama_up else 500)
    monkeypatch.setattr(capacity, "_gpus", lambda: [])
    result = probe(Settings(ollama_num_ctx="auto", ollama_num_ctx_fallback=4096))
    assert (result.window, result.source) == (4096, "fallback")
    assert result.model_limit == (40960 if ollama_up else None)


def test_a_configured_window_is_kept_but_limits_are_still_reported(monkeypatch) -> None:
    fake_ollama(monkeypatch)
    monkeypatch.setattr(capacity, "_gpus", lambda: [_Gpu("RTX", RTX_4070_8GB, 0)])
    result = probe(Settings(ollama_num_ctx=8192))
    assert (result.window, result.source, result.vram_limit) == (8192, "config", 6144)
