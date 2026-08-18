from unittest.mock import AsyncMock

import pytest

import reranker_service


@pytest.fixture(autouse=True)
def _restore_settings(monkeypatch):
    monkeypatch.setattr(reranker_service._settings, "model_id", "Qwen/Qwen3-Reranker-0.6B")
    monkeypatch.setattr(reranker_service._settings, "model_revision", None)
    monkeypatch.setattr(reranker_service._settings, "dtype", "float16")
    monkeypatch.setattr(reranker_service._settings, "max_model_len", 2048)
    monkeypatch.setattr(reranker_service._settings, "max_num_seqs", 1)
    monkeypatch.setattr(reranker_service._settings, "max_num_batched_tokens", 2048)
    monkeypatch.setattr(reranker_service._settings, "gpu_memory_utilization", 0.08)
    monkeypatch.setattr(reranker_service._settings, "quantization", "none")


def test_build_vllm_command_uses_fixed_pooling_limits():
    command = reranker_service._build_vllm_command()

    assert command[command.index("--runner") + 1] == "pooling"
    assert command[command.index("--convert") + 1] == "classify"
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.08"
    assert "--enforce-eager" in command
    assert "--chat-template" in command
    assert "--hf-overrides" in command
    assert "--task" not in command
    assert "--disable-frontend-multiprocessing" not in command
    assert "--quantization" not in command


def test_build_vllm_command_bitsandbytes_is_explicit(monkeypatch):
    monkeypatch.setattr(reranker_service._settings, "quantization", "bitsandbytes")

    command = reranker_service._build_vllm_command()

    assert command[command.index("--quantization") + 1] == "bitsandbytes"


def test_prepare_rerank_payload_defaults_to_all_documents():
    payload, documents = reranker_service.prepare_rerank_payload(
        {"query": "capital of China", "documents": ["Beijing", "Paris"]}
    )

    assert documents == ["Beijing", "Paris"]
    assert payload["top_n"] == 2
    assert payload["model"] == "Qwen/Qwen3-Reranker-0.6B"


@pytest.mark.parametrize(
    "request_payload,match",
    [
        ({"query": "", "documents": ["x"]}, "query"),
        ({"query": "q", "documents": []}, "documents"),
        ({"query": "q", "documents": [" "]}, "documents\\[0\\]"),
        ({"query": "q", "documents": ["x"], "top_n": 2}, "top_n"),
        (
            {"query": "q", "documents": ["x"], "model": "another/model"},
            "does not match",
        ),
    ],
)
def test_prepare_rerank_payload_rejects_invalid_input(request_payload, match):
    with pytest.raises(reranker_service.InputValidationError, match=match):
        reranker_service.prepare_rerank_payload(request_payload)


def test_prepare_rerank_payload_rejects_more_than_50_documents():
    with pytest.raises(reranker_service.InputValidationError, match="at most 50"):
        reranker_service.prepare_rerank_payload(
            {"query": "q", "documents": [f"document-{index}" for index in range(51)]}
        )


def test_normalize_response_sorts_scores_and_preserves_original_index():
    payload = reranker_service._normalize_response(
        {
            "id": "rerank-test",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
            ],
        },
        ["irrelevant", "Beijing is China's capital"],
        1,
    )

    assert payload["results"] == [
        {
            "index": 1,
            "document": {"text": "Beijing is China's capital"},
            "relevance_score": 0.9,
        }
    ]


def test_stop_terminates_process_group_after_api_parent_exits(monkeypatch):
    process = object()
    terminated = []
    monkeypatch.setattr(reranker_service, "_process", process)
    monkeypatch.setattr(reranker_service, "terminate_process_group", terminated.append)

    reranker_service._stop_locked()

    assert terminated == [process]
    assert reranker_service._process is None


@pytest.mark.anyio
async def test_reload_only_restarts_reranker(monkeypatch):
    stop = AsyncMock()
    monkeypatch.setattr(reranker_service, "shutdown_reranker", stop)
    monkeypatch.setattr(reranker_service, "_stop_locked", lambda: None)
    monkeypatch.setattr(reranker_service, "_start_locked", lambda: None)
    monkeypatch.setattr(reranker_service, "wait_for_reranker_ready", AsyncMock())
    monkeypatch.setattr(
        reranker_service,
        "get_reranker_health_payload",
        AsyncMock(return_value={"ready": True, "quantization": "bitsandbytes"}),
    )

    result = await reranker_service.reload_reranker({"quantization": "bitsandbytes"})

    assert result["ready"] is True
    assert reranker_service._settings.quantization == "bitsandbytes"
    assert reranker_service._settings.gpu_memory_utilization == (
        reranker_service.RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION
    )
    stop.assert_not_awaited()


@pytest.mark.anyio
async def test_preload_uses_quantized_memory_limit_for_4bit_fallback(monkeypatch):
    attempts = []

    async def capture_attempt(*_, **__):
        attempts.append(
            (
                reranker_service._settings.gpu_memory_utilization,
                reranker_service._settings.quantization,
            )
        )
        if len(attempts) < 3:
            raise reranker_service.BackendUnavailableError("startup failed")

    monkeypatch.setattr(reranker_service, "ensure_reranker_started", capture_attempt)
    monkeypatch.setattr(reranker_service, "_stop_locked", lambda: None)
    monkeypatch.setattr(reranker_service, "RERANKER_PRELOAD_RETRY_DELAY", 0)

    await reranker_service.maybe_preload_reranker()

    assert attempts == [
        (0.08, "none"),
        (reranker_service.RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION, "none"),
        (reranker_service.RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION, "bitsandbytes"),
    ]


@pytest.mark.anyio
async def test_preload_retries_configured_4bit_after_transient_failure(monkeypatch):
    attempts = []

    async def capture_attempt(*_, **__):
        attempts.append(
            (
                reranker_service._settings.gpu_memory_utilization,
                reranker_service._settings.quantization,
            )
        )
        if len(attempts) == 1:
            raise reranker_service.BackendUnavailableError("transient profiling race")

    monkeypatch.setattr(reranker_service._settings, "quantization", "bitsandbytes")
    monkeypatch.setattr(
        reranker_service._settings,
        "gpu_memory_utilization",
        reranker_service.RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION,
    )
    monkeypatch.setattr(reranker_service, "ensure_reranker_started", capture_attempt)
    monkeypatch.setattr(reranker_service, "_stop_locked", lambda: None)
    monkeypatch.setattr(reranker_service, "RERANKER_PRELOAD_RETRY_DELAY", 0)

    await reranker_service.maybe_preload_reranker()

    assert attempts == [
        (reranker_service.RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION, "bitsandbytes"),
        (reranker_service.RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION, "bitsandbytes"),
    ]
