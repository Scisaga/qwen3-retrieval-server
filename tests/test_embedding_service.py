import pytest

import embedding_service


class _CompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture(autouse=True)
def _configured_model_dimensions(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "max_dimensions", 2560)


def test_extract_max_dimensions_from_qwen_hidden_size():
    assert embedding_service._extract_max_dimensions({"hidden_size": 2560}) == 2560


def test_extract_max_dimensions_prefers_projection_dimension():
    assert embedding_service._extract_max_dimensions({"projection_dim": 1024, "hidden_size": 2560}) == 1024


def test_prepare_backend_payload_for_single_document():
    payload = embedding_service.prepare_backend_payload(
        {
            "input": "The capital of China is Beijing.",
            "dimensions": 512,
        }
    )

    assert payload["input"] == "The capital of China is Beijing."
    assert payload["dimensions"] == 512
    assert payload["model"] == embedding_service.get_current_model_id()


def test_prepare_backend_payload_for_query_uses_default_instruction(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "default_query_instruction", "default task")

    payload = embedding_service.prepare_backend_payload(
        {
            "input": ["What is the capital of China?"],
            "input_type": "query",
        }
    )

    assert payload["input"] == ["Instruct: default task\nQuery:What is the capital of China?"]


def test_prepare_backend_payload_for_query_uses_custom_instruction():
    payload = embedding_service.prepare_backend_payload(
        {
            "input": "Explain gravity",
            "input_type": "query",
            "instruction": "Answer with retrieval-friendly vectors",
        }
    )

    assert payload["input"] == "Instruct: Answer with retrieval-friendly vectors\nQuery:Explain gravity"


def test_prepare_backend_payload_rejects_invalid_dimensions():
    with pytest.raises(embedding_service.InputValidationError):
        embedding_service.prepare_backend_payload({"input": "hello", "dimensions": 31})

    with pytest.raises(embedding_service.InputValidationError, match="2560"):
        embedding_service.prepare_backend_payload({"input": "hello", "dimensions": 2561})


def test_prepare_backend_payload_rejects_invalid_input_type():
    with pytest.raises(embedding_service.InputValidationError):
        embedding_service.prepare_backend_payload({"input": "hello", "input_type": "mixed"})


def test_build_backend_env_strips_vllm_port(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "8001")

    backend_env = embedding_service._build_backend_env()

    assert "VLLM_PORT" not in backend_env


def test_build_backend_env_sets_cuda_visible_devices(monkeypatch):
    backend_env = embedding_service._build_backend_env("GPU-test-0")

    assert backend_env["CUDA_VISIBLE_DEVICES"] == "GPU-test-0"


def test_detect_visible_gpu_identifiers_returns_cuda_ordinals_for_nvidia_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        return _CompletedProcess(
            stdout=(
                "GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-87a92d2a-6521-1e36-1ca9-21fd8085b365)\n"
                "GPU 1: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-95b5895b-c5f3-440b-4bd2-7160ec4da0d1)\n"
            )
        )

    monkeypatch.setattr(embedding_service.subprocess, "run", fake_run)

    identifiers = embedding_service._detect_visible_gpu_identifiers()

    assert identifiers == ["0", "1"]


def test_detect_visible_gpu_identifiers_converts_uuid_env_to_ordinals(monkeypatch):
    def fake_run(*args, **kwargs):
        return _CompletedProcess(returncode=1)

    monkeypatch.setattr(embedding_service.subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(
        "NVIDIA_VISIBLE_DEVICES",
        "GPU-87a92d2a-6521-1e36-1ca9-21fd8085b365,GPU-95b5895b-c5f3-440b-4bd2-7160ec4da0d1",
    )

    identifiers = embedding_service._detect_visible_gpu_identifiers()

    assert identifiers == ["0", "1"]


def test_validate_backend_settings_rejects_batched_tokens_smaller_than_max_model_len(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "max_model_len", 4096)
    monkeypatch.setattr(
        embedding_service._settings,
        "extra_args",
        "--enforce-eager --max-num-batched-tokens 1024",
    )

    with pytest.raises(embedding_service.BackendUnavailableError):
        embedding_service._validate_backend_settings()


def test_build_vllm_command_auto_enables_qwen3_matryoshka(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "model_id", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(embedding_service._settings, "extra_args", "--enforce-eager")

    command = embedding_service._build_vllm_command()

    assert "--hf_overrides" in command
    override_index = command.index("--hf_overrides")
    assert command[override_index + 1] == '{"is_matryoshka": true}'


def test_build_vllm_command_does_not_duplicate_existing_hf_overrides(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "model_id", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(
        embedding_service._settings,
        "extra_args",
        '--hf_overrides {"matryoshka_dimensions":[1024]}',
    )

    command = embedding_service._build_vllm_command()

    assert command.count("--hf_overrides") == 1


def test_build_vllm_command_leaves_non_qwen_models_unchanged(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "model_id", "BAAI/bge-m3")
    monkeypatch.setattr(embedding_service._settings, "extra_args", "--enforce-eager")

    command = embedding_service._build_vllm_command()

    assert "--hf_overrides" not in command


def test_build_backend_replicas_layout_uses_one_replica_per_visible_gpu(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "manage_backend_process", True)
    monkeypatch.setattr(embedding_service._settings, "backend_port", 8001)
    monkeypatch.setattr(embedding_service._settings, "backend_host", "127.0.0.1")
    monkeypatch.setattr(embedding_service._settings, "extra_args", "--enforce-eager")
    monkeypatch.setattr(embedding_service, "_detect_visible_gpu_identifiers", lambda: ["GPU-a", "GPU-b"])
    monkeypatch.delenv("BACKEND_REPLICA_COUNT", raising=False)
    monkeypatch.delenv("AUTO_BACKEND_REPLICAS", raising=False)

    layout = embedding_service._build_backend_replicas_layout()

    assert [replica.port for replica in layout] == [8001, 8002]
    assert [replica.device_identifier for replica in layout] == ["GPU-a", "GPU-b"]


def test_build_backend_replicas_layout_keeps_single_backend_for_tensor_parallel(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "manage_backend_process", True)
    monkeypatch.setattr(embedding_service._settings, "extra_args", "--tensor-parallel-size 2")
    monkeypatch.setattr(embedding_service, "_detect_visible_gpu_identifiers", lambda: ["GPU-a", "GPU-b"])
    monkeypatch.delenv("BACKEND_REPLICA_COUNT", raising=False)

    layout = embedding_service._build_backend_replicas_layout()

    assert len(layout) == 1
    assert layout[0].device_identifier is None


def test_validate_backend_settings_rejects_replica_override_with_tensor_parallel(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "manage_backend_process", True)
    monkeypatch.setattr(embedding_service._settings, "extra_args", "--tensor-parallel-size 2")
    monkeypatch.setenv("BACKEND_REPLICA_COUNT", "2")

    with pytest.raises(embedding_service.BackendUnavailableError):
        embedding_service._validate_backend_settings()


def test_ordered_backend_candidates_round_robin_ready_replicas(monkeypatch):
    replicas = [
        embedding_service.BackendReplica(replica_index=0, port=8001, base_url="http://127.0.0.1:8001", ready=True),
        embedding_service.BackendReplica(replica_index=1, port=8002, base_url="http://127.0.0.1:8002", ready=True),
    ]
    monkeypatch.setattr(embedding_service, "_backend_replicas", replicas)
    monkeypatch.setattr(embedding_service, "_backend_router_index", 0)

    first = embedding_service._ordered_backend_candidates_locked()
    second = embedding_service._ordered_backend_candidates_locked()

    assert [replica.replica_index for replica in first] == [0, 1]
    assert [replica.replica_index for replica in second] == [1, 0]


@pytest.mark.anyio
async def test_reload_backend_refreshes_model_max_dimensions(monkeypatch):
    async def fake_wait_for_backend_ready(timeout_s=None):
        return None

    async def fake_get_health_payload():
        return {"model_id": embedding_service._settings.model_id, "max_dimensions": 1024}

    monkeypatch.setattr(embedding_service._settings, "model_id", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(embedding_service, "resolve_model_max_dimensions", lambda *args: 1024)
    monkeypatch.setattr(embedding_service, "_stop_backend_process_locked", lambda: None)
    monkeypatch.setattr(embedding_service, "_start_backend_process_locked", lambda: None)
    monkeypatch.setattr(embedding_service, "wait_for_backend_ready", fake_wait_for_backend_ready)
    monkeypatch.setattr(embedding_service, "get_health_payload", fake_get_health_payload)

    payload = await embedding_service.reload_backend({"model_id": "Qwen/Qwen3-Embedding-0.6B"})

    assert embedding_service._settings.max_dimensions == 1024
    assert payload["max_dimensions"] == 1024


@pytest.mark.anyio
async def test_probe_backend_health_treats_empty_200_response_as_ready(monkeypatch):
    replicas = [
        embedding_service.BackendReplica(replica_index=0, port=8001, base_url="http://127.0.0.1:8001"),
        embedding_service.BackendReplica(replica_index=1, port=8002, base_url="http://127.0.0.1:8002"),
    ]

    async def fake_probe_single_backend_health(client, base_url):
        return True, None, "", "/health"

    monkeypatch.setattr(embedding_service._settings, "manage_backend_process", True)
    monkeypatch.setattr(embedding_service, "_backend_replicas", replicas)
    monkeypatch.setattr(embedding_service, "_probe_single_backend_health", fake_probe_single_backend_health)

    healthy, payload, message = await embedding_service._probe_backend_health()

    assert healthy is True
    assert payload is None
    assert message == ""
    assert all(replica.ready for replica in replicas)
