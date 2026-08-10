import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from huggingface_hub import hf_hub_download

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-4B")
MODEL_REVISION = os.getenv("MODEL_REVISION")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "12302"))
BACKEND_HOST = os.getenv("BACKEND_HOST") or os.getenv("VLLM_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT") or os.getenv("VLLM_PORT", "8001"))
HF_HOME = os.getenv("HF_HOME", "/models")
DTYPE = os.getenv("DTYPE", "float16")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.72"))
DEFAULT_QUERY_INSTRUCTION = os.getenv(
    "DEFAULT_QUERY_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
BACKEND_START_TIMEOUT = int(os.getenv("BACKEND_START_TIMEOUT", "600"))
BACKEND_HTTP_TIMEOUT = float(os.getenv("BACKEND_HTTP_TIMEOUT", "120"))
BACKEND_POLL_INTERVAL = float(os.getenv("BACKEND_POLL_INTERVAL", "1.0"))
REQUEST_READY_TIMEOUT = float(os.getenv("REQUEST_READY_TIMEOUT", "3.0"))
BACKEND_PORT_SCAN_WINDOW = int(os.getenv("BACKEND_PORT_SCAN_WINDOW", "0"))
VLLM_EXTRA_ARGS = os.getenv("VLLM_EXTRA_ARGS", "")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").strip()
MANAGE_BACKEND_PROCESS = (
    os.getenv("MANAGE_BACKEND_PROCESS", "0" if VLLM_BASE_URL else "1").strip().lower()
    not in ("0", "false", "no", "off")
)
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1").strip().lower() not in ("0", "false", "no", "off")

_DEFAULT_BACKEND_PATH = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
_AUTO_BACKEND_REPLICAS_ENV = "AUTO_BACKEND_REPLICAS"
_BACKEND_REPLICA_COUNT_ENV = "BACKEND_REPLICA_COUNT"
_MODEL_PARALLEL_FLAG_ALIASES = (
    ("--tensor-parallel-size", "--tensor_parallel_size"),
    ("--pipeline-parallel-size", "--pipeline_parallel_size"),
)
_MATRYOSHKA_OVERRIDE_FLAG_ALIASES = ("--hf_overrides", "--hf-overrides")
_backend_lock = threading.RLock()
_backend_replicas: list["BackendReplica"] = []
_backend_router_index = 0
_backend_started_at: Optional[float] = None
_backend_ready = False
_backend_last_error = ""
_backend_probe_path = ""


class InputValidationError(ValueError):
    pass


class BackendUnavailableError(RuntimeError):
    pass


class BackendProxyError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, payload: Optional[dict[str, Any]] = None):
        self.status_code = status_code
        self.payload = payload
        super().__init__(message)


@dataclass
class BackendSettings:
    model_id: str = MODEL_ID
    model_revision: Optional[str] = MODEL_REVISION
    backend_host: str = BACKEND_HOST
    backend_port: int = BACKEND_PORT
    dtype: str = DTYPE
    max_model_len: int = MAX_MODEL_LEN
    max_dimensions: Optional[int] = None
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    default_query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    hf_home: str = HF_HOME
    manage_backend_process: bool = MANAGE_BACKEND_PROCESS
    preload_model: bool = PRELOAD_MODEL
    extra_args: str = VLLM_EXTRA_ARGS
    backend_base_url: str = VLLM_BASE_URL or _DEFAULT_BACKEND_PATH


@dataclass
class BackendReplica:
    replica_index: int
    port: int
    base_url: str
    device_identifier: Optional[str] = None
    process: Optional[subprocess.Popen[Any]] = None
    started_at: Optional[float] = None
    ready: bool = False
    last_error: str = ""
    probe_path: str = ""
    health: Optional[dict[str, Any]] = None


_settings = BackendSettings()


def _extract_max_dimensions(model_config: dict[str, Any]) -> int:
    dimension_paths = (
        ("sentence_embedding_dimension",),
        ("embedding_dimension",),
        ("projection_dim",),
        ("text_config", "projection_dim"),
        ("hidden_size",),
        ("text_config", "hidden_size"),
        ("d_model",),
        ("text_config", "d_model"),
    )
    for path in dimension_paths:
        value: Any = model_config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value

    raise BackendUnavailableError(
        "Model config does not expose a supported embedding dimension field "
        "(sentence_embedding_dimension, embedding_dimension, projection_dim, hidden_size, or d_model)."
    )


def _load_model_config(model_id: str, model_revision: Optional[str], hf_home: str) -> dict[str, Any]:
    local_config_path = Path(model_id).expanduser() / "config.json"
    if local_config_path.is_file():
        config_path = local_config_path
    else:
        download_args: dict[str, Any] = {
            "repo_id": model_id,
            "filename": "config.json",
            "revision": model_revision,
            "cache_dir": hf_home,
        }
        try:
            config_path = Path(hf_hub_download(**download_args, local_files_only=True))
        except Exception:
            try:
                config_path = Path(hf_hub_download(**download_args))
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Unable to load model config for {model_id!r}: {exc}"
                ) from exc

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except (OSError, ValueError) as exc:
        raise BackendUnavailableError(
            f"Unable to read model config for {model_id!r} from {config_path}: {exc}"
        ) from exc
    if not isinstance(model_config, dict):
        raise BackendUnavailableError(f"Invalid model config for {model_id!r}: expected a JSON object.")
    return model_config


def resolve_model_max_dimensions(
    model_id: str,
    model_revision: Optional[str] = None,
    hf_home: str = HF_HOME,
) -> int:
    return _extract_max_dimensions(_load_model_config(model_id, model_revision, hf_home))


def _ensure_max_dimensions_locked() -> int:
    if _settings.max_dimensions is None:
        _settings.max_dimensions = resolve_model_max_dimensions(
            _settings.model_id,
            _settings.model_revision,
            _settings.hf_home,
        )
    return _settings.max_dimensions


def get_max_dimensions() -> int:
    with _backend_lock:
        return _ensure_max_dimensions_locked()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def _apply_proxy_env(target_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = target_env if target_env is not None else os.environ
    http_proxy = (env.get("HTTP_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
    https_proxy = (env.get("HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or "").strip()
    no_proxy = (env.get("NO_PROXY") or os.getenv("NO_PROXY") or "").strip()

    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
        if not https_proxy:
            https_proxy = http_proxy
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy

    env["NO_PROXY"] = no_proxy or "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def _build_backend_env(device_identifier: Optional[str] = None) -> dict[str, str]:
    env = _apply_proxy_env(dict(os.environ))
    env["HF_HOME"] = _settings.hf_home
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    # Our wrapper used to expose BACKEND_PORT through the env name `VLLM_PORT`.
    # That collides with vLLM's own runtime env var for internal IPC port
    # allocation, which triggers misleading logs like "Port 8001 is already in
    # use, trying port 8002" after the API server has already bound 8001.
    env.pop("VLLM_PORT", None)
    if device_identifier is not None:
        env["CUDA_VISIBLE_DEVICES"] = device_identifier
    return env


def _get_extra_arg_value(flag: str, extra_args: str) -> Optional[str]:
    tokens = shlex.split(extra_args or "")
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _get_first_extra_arg_value(flags: tuple[str, ...], extra_args: str) -> Optional[str]:
    for flag in flags:
        value = _get_extra_arg_value(flag, extra_args)
        if value is not None:
            return value
    return None


def _has_extra_arg(flags: tuple[str, ...], extra_args: str) -> bool:
    tokens = shlex.split(extra_args or "")
    for token in tokens:
        if token in flags:
            return True
        for flag in flags:
            if token.startswith(f"{flag}="):
                return True
    return False


def _requested_model_parallelism(extra_args: str) -> bool:
    for aliases in _MODEL_PARALLEL_FLAG_ALIASES:
        raw_value = _get_first_extra_arg_value(aliases, extra_args)
        if raw_value is None:
            continue
        try:
            if int(raw_value) > 1:
                return True
        except ValueError:
            return True
    return False


def _should_enable_qwen3_matryoshka_override(model_id: str, extra_args: str) -> bool:
    normalized_model_id = (model_id or "").strip().lower()
    if not normalized_model_id.startswith("qwen/qwen3-embedding-"):
        return False
    if _has_extra_arg(_MATRYOSHKA_OVERRIDE_FLAG_ALIASES, extra_args):
        return False
    return True


def _backend_replica_count_override() -> Optional[int]:
    raw_value = os.getenv(_BACKEND_REPLICA_COUNT_ENV, "").strip()
    if not raw_value:
        return None
    try:
        count = int(raw_value)
    except ValueError as exc:
        raise BackendUnavailableError(
            f"Invalid {_BACKEND_REPLICA_COUNT_ENV}: expected positive integer, got {raw_value!r}."
        ) from exc
    if count < 1:
        raise BackendUnavailableError(
            f"Invalid {_BACKEND_REPLICA_COUNT_ENV}: expected positive integer, got {raw_value!r}."
        )
    return count


def _detect_visible_gpu_identifiers() -> list[str]:
    try:
        probe_env = _apply_proxy_env(dict(os.environ))
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=probe_env,
        )
        if result.returncode == 0:
            gpu_count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))
            if gpu_count > 0:
                return [str(index) for index in range(gpu_count)]
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        lowered = raw_value.lower()
        if lowered in ("none", "void"):
            return []
        if lowered == "all":
            continue
        identifiers = [token.strip() for token in raw_value.split(",") if token.strip()]
        if identifiers:
            if any(identifier.startswith("GPU-") for identifier in identifiers):
                return [str(index) for index in range(len(identifiers))]
            return identifiers
    return []


def _desired_backend_device_identifiers() -> list[Optional[str]]:
    if not _settings.manage_backend_process:
        return [None]

    requested_parallelism = _requested_model_parallelism(_settings.extra_args)
    replica_override = _backend_replica_count_override()
    if requested_parallelism:
        if replica_override is not None and replica_override > 1:
            raise BackendUnavailableError(
                f"{_BACKEND_REPLICA_COUNT_ENV} cannot exceed 1 when tensor/pipeline parallelism is enabled."
            )
        return [None]

    if replica_override == 1:
        return [None]
    if replica_override is None and not _env_flag(_AUTO_BACKEND_REPLICAS_ENV, "1"):
        return [None]

    identifiers = _detect_visible_gpu_identifiers()
    if replica_override is not None:
        if identifiers and replica_override > len(identifiers):
            raise BackendUnavailableError(
                f"{_BACKEND_REPLICA_COUNT_ENV}={replica_override} exceeds visible GPU count ({len(identifiers)})."
            )
        if identifiers:
            return identifiers[:replica_override]
        return [str(index) for index in range(replica_override)]

    if len(identifiers) <= 1:
        return [None]
    return identifiers


def _build_backend_replicas_layout() -> list[BackendReplica]:
    device_identifiers = _desired_backend_device_identifiers()
    return [
        BackendReplica(
            replica_index=index,
            port=_settings.backend_port + index,
            base_url=f"http://{_settings.backend_host}:{_settings.backend_port + index}",
            device_identifier=device_identifier,
        )
        for index, device_identifier in enumerate(device_identifiers)
    ]


def _same_replica_layout(current: list[BackendReplica], desired: list[BackendReplica]) -> bool:
    if len(current) != len(desired):
        return False
    for current_replica, desired_replica in zip(current, desired):
        if current_replica.port != desired_replica.port:
            return False
        if current_replica.device_identifier != desired_replica.device_identifier:
            return False
    return True


def _replica_alive(replica: BackendReplica) -> bool:
    return replica.process is not None and replica.process.poll() is None


def _refresh_backend_summary_locked() -> None:
    global _backend_ready, _backend_last_error, _backend_probe_path, _backend_started_at

    if not _settings.manage_backend_process:
        return

    ready_replicas = [replica for replica in _backend_replicas if replica.ready]
    _backend_ready = bool(ready_replicas)
    _backend_probe_path = next(
        (replica.probe_path for replica in ready_replicas if replica.probe_path),
        next((replica.probe_path for replica in _backend_replicas if replica.probe_path), ""),
    )
    if ready_replicas:
        _backend_last_error = ""
        _settings.backend_base_url = ready_replicas[0].base_url
    else:
        _backend_last_error = next((replica.last_error for replica in _backend_replicas if replica.last_error), "")
        if _backend_replicas:
            _settings.backend_base_url = _backend_replicas[0].base_url

    started_at_values = [replica.started_at for replica in _backend_replicas if replica.started_at is not None]
    _backend_started_at = min(started_at_values) if started_at_values else None


def _ensure_backend_layout_locked() -> None:
    global _backend_replicas, _backend_router_index

    if not _settings.manage_backend_process:
        _backend_replicas = []
        _backend_router_index = 0
        return

    desired_layout = _build_backend_replicas_layout()
    if _same_replica_layout(_backend_replicas, desired_layout):
        return

    _backend_replicas = desired_layout
    _backend_router_index = 0
    _refresh_backend_summary_locked()


def _validate_backend_settings() -> None:
    raw_batched_tokens = _get_extra_arg_value("--max-num-batched-tokens", _settings.extra_args)
    if raw_batched_tokens is not None:
        try:
            max_num_batched_tokens = int(raw_batched_tokens)
        except ValueError as exc:
            raise BackendUnavailableError(
                f"Invalid VLLM_EXTRA_ARGS: --max-num-batched-tokens must be an integer, got {raw_batched_tokens!r}."
            ) from exc

        if max_num_batched_tokens < _settings.max_model_len:
            raise BackendUnavailableError(
                "Invalid vLLM configuration: --max-num-batched-tokens "
                f"({max_num_batched_tokens}) must be >= max_model_len ({_settings.max_model_len})."
            )

    _desired_backend_device_identifiers()


def _build_vllm_command(port: Optional[int] = None) -> list[str]:
    extra_args = _settings.extra_args.strip()
    extra_tokens = shlex.split(extra_args) if extra_args else []
    command = [
        "vllm",
        "serve",
        _settings.model_id,
        "--host",
        _settings.backend_host,
        "--port",
        str(port if port is not None else _settings.backend_port),
        "--task",
        "embed",
        "--dtype",
        _settings.dtype,
        "--max-model-len",
        str(_settings.max_model_len),
        "--gpu-memory-utilization",
        str(_settings.gpu_memory_utilization),
        "--served-model-name",
        _settings.model_id,
    ]
    if _settings.model_revision:
        command.extend(["--revision", _settings.model_revision])
    if _env_flag("TRUST_REMOTE_CODE", "0"):
        command.append("--trust-remote-code")
    if _should_enable_qwen3_matryoshka_override(_settings.model_id, extra_args):
        command.extend(["--hf_overrides", json.dumps({"is_matryoshka": True})])
    if extra_tokens:
        command.extend(extra_tokens)
    return command


def _backend_alive() -> bool:
    return any(_replica_alive(replica) for replica in _backend_replicas)


def _mark_backend_not_ready(message: str = "") -> None:
    global _backend_ready, _backend_last_error
    _backend_ready = False
    if message:
        _backend_last_error = message


def _backend_state(healthy: bool, process_alive: bool, exit_code: Optional[int]) -> str:
    if healthy:
        return "ready"
    if process_alive:
        return "starting"
    if exit_code is not None:
        return "exited"
    return "stopped"


def _backend_message(
    healthy: bool,
    process_alive: bool,
    exit_code: Optional[int],
    probe_message: str,
) -> str:
    if healthy:
        return ""
    if process_alive:
        return (
            "vLLM 进程已启动，但健康检查尚未就绪。"
            "这通常表示模型仍在加载权重到 GPU，或内部 engine 仍在初始化。"
        )
    if exit_code is not None:
        return _backend_last_error or f"vLLM exited with code {exit_code}."
    return _backend_last_error or probe_message or "Backend is not reachable."


def _start_backend_process_locked() -> None:
    global _backend_last_error

    _ensure_max_dimensions_locked()
    if not _settings.manage_backend_process:
        return

    _validate_backend_settings()
    _ensure_backend_layout_locked()

    for replica in _backend_replicas:
        if _replica_alive(replica):
            continue

        command = _build_vllm_command(port=replica.port)
        try:
            replica.process = subprocess.Popen(
                command,
                env=_build_backend_env(replica.device_identifier),
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            replica.process = None
            replica.ready = False
            replica.last_error = "Failed to start vLLM: `vllm` command not found."
            _backend_last_error = replica.last_error
            raise BackendUnavailableError(_backend_last_error) from exc

        replica.started_at = time.time()
        replica.ready = False
        replica.last_error = ""
        replica.probe_path = ""
        replica.health = None

    _backend_last_error = ""
    _refresh_backend_summary_locked()


def _stop_backend_process_locked() -> None:
    for replica in _backend_replicas:
        proc = replica.process
        replica.process = None
        replica.ready = False
        replica.health = None
        replica.probe_path = ""

        if proc is None:
            continue
        if proc.poll() is not None:
            continue

        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    _refresh_backend_summary_locked()


async def _probe_single_backend_health(
    client: httpx.AsyncClient,
    base_url: str,
) -> tuple[bool, Optional[dict[str, Any]], str, str]:
    probe_paths = ["/health", "/v1/models"]
    last_message = ""
    for path in probe_paths:
        try:
            response = await client.get(f"{base_url}{path}")
        except httpx.HTTPError as exc:
            last_message = str(exc)
            continue

        if response.status_code == 404:
            last_message = f"{path} returned HTTP 404 on {base_url}"
            continue
        if response.status_code >= 400:
            last_message = f"{path} returned HTTP {response.status_code} on {base_url}"
            continue

        payload: Optional[dict[str, Any]]
        try:
            payload = response.json()
        except ValueError:
            payload = None

        return True, payload, "", path

    return False, None, last_message or "All connection attempts failed", ""


def _probe_base_urls() -> list[str]:
    base_urls = [_settings.backend_base_url.rstrip("/")]
    if BACKEND_PORT_SCAN_WINDOW <= 0 or _settings.backend_base_url != _DEFAULT_BACKEND_PATH:
        return base_urls

    for offset in range(1, max(0, BACKEND_PORT_SCAN_WINDOW) + 1):
        base_urls.append(f"http://{_settings.backend_host}:{_settings.backend_port + offset}")
    return base_urls


async def _probe_backend_health() -> tuple[bool, Optional[dict[str, Any]], str]:
    global _backend_probe_path
    timeout = httpx.Timeout(3.0, connect=1.0)

    if not _settings.manage_backend_process:
        last_message = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            for base_url in _probe_base_urls():
                healthy, payload, message, probe_path = await _probe_single_backend_health(client, base_url)
                if healthy:
                    _settings.backend_base_url = base_url
                    _backend_probe_path = probe_path
                    return True, payload, ""
                last_message = message

        _backend_probe_path = ""
        return False, None, last_message or "All connection attempts failed"

    async with httpx.AsyncClient(timeout=timeout) as client:
        results: list[tuple[BackendReplica, bool, Optional[dict[str, Any]], str, str]] = []
        for replica in list(_backend_replicas):
            healthy, payload, message, probe_path = await _probe_single_backend_health(client, replica.base_url)
            results.append((replica, healthy, payload, message, probe_path))

    any_healthy = False
    healthy_payload: Optional[dict[str, Any]] = None
    healthy_probe_path = ""
    first_error = ""
    for replica, healthy, payload, message, probe_path in results:
        replica.ready = healthy
        replica.health = payload if healthy else None
        replica.probe_path = probe_path
        if healthy:
            replica.last_error = ""
            if not any_healthy:
                healthy_payload = payload
                healthy_probe_path = probe_path
            any_healthy = True
        elif message:
            replica.last_error = message
            if not first_error:
                first_error = message

    _refresh_backend_summary_locked()
    _backend_probe_path = healthy_probe_path or next(
        (replica.probe_path for replica in _backend_replicas if replica.probe_path),
        "",
    )

    return any_healthy, healthy_payload, "" if any_healthy else first_error or "All connection attempts failed"


async def wait_for_backend_ready(timeout_s: Optional[float] = None) -> None:
    global _backend_ready, _backend_last_error

    if not _settings.manage_backend_process:
        healthy, _, message = await _probe_backend_health()
        _backend_ready = healthy
        if not healthy:
            _backend_last_error = message or "Backend is not reachable."
            raise BackendUnavailableError(_backend_last_error)
        return

    deadline = time.monotonic() + float(timeout_s or BACKEND_START_TIMEOUT)
    while time.monotonic() < deadline:
        with _backend_lock:
            replicas = list(_backend_replicas)

        if not replicas:
            _mark_backend_not_ready("Backend process is not configured.")
            raise BackendUnavailableError(_backend_last_error)

        alive_replicas = [replica for replica in replicas if _replica_alive(replica)]
        if not alive_replicas:
            exited_replica = next(
                (replica for replica in replicas if replica.process is not None and replica.process.poll() is not None),
                None,
            )
            if exited_replica is not None and exited_replica.process is not None:
                _mark_backend_not_ready(
                    f"vLLM replica {exited_replica.replica_index} exited with code {exited_replica.process.returncode}."
                )
            else:
                _mark_backend_not_ready("Backend process is not running.")
            raise BackendUnavailableError(_backend_last_error)

        healthy, _, message = await _probe_backend_health()
        if healthy:
            _backend_ready = True
            _backend_last_error = ""
            return

        _backend_ready = False
        if message:
            _backend_last_error = message
        await asyncio.sleep(BACKEND_POLL_INTERVAL)

    _backend_ready = False
    _backend_last_error = f"Timed out after {int(timeout_s or BACKEND_START_TIMEOUT)}s waiting for vLLM."
    raise BackendUnavailableError(_backend_last_error)


async def ensure_backend_started(wait_ready: bool = True, timeout_s: Optional[float] = None) -> None:
    with _backend_lock:
        _start_backend_process_locked()
    if wait_ready:
        await wait_for_backend_ready(timeout_s=timeout_s)


async def maybe_preload_backend() -> None:
    if not _settings.preload_model:
        return
    try:
        await ensure_backend_started(wait_ready=True)
    except Exception as exc:
        _mark_backend_not_ready(str(exc))


async def shutdown_backend() -> None:
    with _backend_lock:
        _stop_backend_process_locked()


def get_current_model_id() -> str:
    return _settings.model_id


def get_current_settings() -> dict[str, Any]:
    snapshot = asdict(_settings)
    snapshot["public_host"] = HOST
    snapshot["public_port"] = PORT
    snapshot["auto_backend_replicas"] = _env_flag(_AUTO_BACKEND_REPLICAS_ENV, "1")
    snapshot["backend_replica_count"] = len(_build_backend_replicas_layout()) if _settings.manage_backend_process else 1
    return snapshot


def _normalize_text_list(raw_input: Any) -> tuple[list[str], bool]:
    if isinstance(raw_input, str):
        if not raw_input.strip():
            raise InputValidationError("`input` must not be empty.")
        return [raw_input], False

    if isinstance(raw_input, list):
        if not raw_input:
            raise InputValidationError("`input` array must not be empty.")
        normalized: list[str] = []
        for index, item in enumerate(raw_input):
            if not isinstance(item, str):
                raise InputValidationError(f"`input[{index}]` must be a string.")
            if not item.strip():
                raise InputValidationError(f"`input[{index}]` must not be empty.")
            normalized.append(item)
        return normalized, True

    raise InputValidationError("`input` must be a string or an array of strings.")


def _validate_dimensions(dimensions: Optional[int]) -> Optional[int]:
    if dimensions is None:
        return None
    if not isinstance(dimensions, int):
        raise InputValidationError("`dimensions` must be an integer.")
    max_dimensions = get_max_dimensions()
    if dimensions < 32 or dimensions > max_dimensions:
        raise InputValidationError(f"`dimensions` must be between 32 and {max_dimensions}.")
    return dimensions


def _validate_input_type(input_type: Optional[str]) -> Optional[str]:
    if input_type is None:
        return None
    if input_type not in ("query", "document"):
        raise InputValidationError("`input_type` must be `query` or `document`.")
    return input_type


def format_query_text(text: str, instruction: str) -> str:
    cleaned_instruction = instruction.strip()
    return f"Instruct: {cleaned_instruction}\nQuery:{text}"


def prepare_backend_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    texts, input_was_list = _normalize_text_list(request_payload.get("input"))
    input_type = _validate_input_type(request_payload.get("input_type"))
    dimensions = _validate_dimensions(request_payload.get("dimensions"))
    instruction = (request_payload.get("instruction") or "").strip() or _settings.default_query_instruction

    prepared_texts = texts
    if input_type == "query":
        prepared_texts = [format_query_text(text, instruction) for text in texts]

    backend_input: str | list[str]
    if input_was_list:
        backend_input = prepared_texts
    else:
        backend_input = prepared_texts[0]

    payload: dict[str, Any] = {
        "input": backend_input,
        "model": request_payload.get("model") or _settings.model_id,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if request_payload.get("encoding_format") is not None:
        payload["encoding_format"] = request_payload["encoding_format"]
    if request_payload.get("user") is not None:
        payload["user"] = request_payload["user"]
    return payload


async def _post_embeddings_to_base_url(
    base_url: str,
    payload: dict[str, Any],
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout_s or BACKEND_HTTP_TIMEOUT, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url.rstrip('/')}/v1/embeddings", json=payload)
    except httpx.HTTPError as exc:
        raise BackendUnavailableError(f"Failed to reach vLLM backend {base_url}: {exc}") from exc

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {
                "error": {
                    "message": response.text or f"Backend returned HTTP {response.status_code}",
                    "type": "backend_error",
                    "param": None,
                    "code": response.status_code,
                }
            }
        raise BackendProxyError(
            f"Backend returned HTTP {response.status_code}",
            status_code=response.status_code,
            payload=error_payload,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BackendProxyError("Backend returned non-JSON response.", status_code=502) from exc


def _ordered_backend_candidates_locked() -> list[BackendReplica]:
    global _backend_router_index

    if not _backend_replicas:
        return []

    ready_replicas = [replica for replica in _backend_replicas if replica.ready]
    candidates = ready_replicas or [replica for replica in _backend_replicas if _replica_alive(replica)] or list(
        _backend_replicas
    )
    start_index = _backend_router_index % len(candidates)
    ordered = candidates[start_index:] + candidates[:start_index]
    _backend_router_index = (_backend_router_index + 1) % len(candidates)
    return ordered


async def _post_embeddings(payload: dict[str, Any], timeout_s: Optional[float] = None) -> dict[str, Any]:
    global _backend_ready, _backend_last_error

    if not _settings.manage_backend_process:
        response_payload = await _post_embeddings_to_base_url(_settings.backend_base_url, payload, timeout_s=timeout_s)
        _backend_ready = True
        _backend_last_error = ""
        return response_payload

    with _backend_lock:
        candidates = _ordered_backend_candidates_locked()

    if not candidates:
        _mark_backend_not_ready("No managed vLLM backends are configured.")
        raise BackendUnavailableError(_backend_last_error)

    last_unavailable: Optional[BackendUnavailableError] = None
    for replica in candidates:
        try:
            response_payload = await _post_embeddings_to_base_url(replica.base_url, payload, timeout_s=timeout_s)
        except BackendUnavailableError as exc:
            replica.ready = False
            replica.last_error = str(exc)
            last_unavailable = exc
            continue

        replica.ready = True
        replica.last_error = ""
        _settings.backend_base_url = replica.base_url
        _backend_ready = True
        _backend_last_error = ""
        return response_payload

    _mark_backend_not_ready(str(last_unavailable) if last_unavailable is not None else "No vLLM backend is reachable.")
    if last_unavailable is not None:
        raise last_unavailable
    raise BackendUnavailableError(_backend_last_error)


async def create_embeddings(request_payload: dict[str, Any]) -> dict[str, Any]:
    global _backend_ready, _backend_last_error
    backend_payload = prepare_backend_payload(request_payload)
    try:
        await ensure_backend_started(wait_ready=True, timeout_s=REQUEST_READY_TIMEOUT)
    except BackendUnavailableError as exc:
        if _backend_alive():
            # Some vLLM versions/flags can make /health lag behind actual readiness.
            # Try one short real request before declaring startup not ready.
            try:
                response_payload = await _post_embeddings(backend_payload, timeout_s=max(15.0, REQUEST_READY_TIMEOUT))
                _backend_ready = True
                _backend_last_error = ""
                return response_payload
            except (BackendUnavailableError, BackendProxyError):
                raise BackendUnavailableError(
                    "Backend is still starting and may still be loading model weights on GPU. "
                    "Please wait a bit and refresh /health."
                ) from exc
        raise
    response_payload = await _post_embeddings(backend_payload)
    _backend_ready = True
    return response_payload


async def embed_texts(
    texts: str | list[str],
    input_type: Optional[str] = None,
    instruction: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"input": texts}
    if input_type is not None:
        payload["input_type"] = input_type
    if instruction is not None:
        payload["instruction"] = instruction
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return await create_embeddings(payload)


def _managed_backend_runtime_locked() -> tuple[Optional[int], Optional[int], bool, int, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for replica in _backend_replicas:
        process_alive = _replica_alive(replica)
        exit_code = replica.process.poll() if replica.process is not None else None
        details.append(
            {
                "replica_index": replica.replica_index,
                "base_url": replica.base_url,
                "port": replica.port,
                "device_identifier": replica.device_identifier,
                "ready": replica.ready,
                "process_alive": process_alive,
                "pid": replica.process.pid if replica.process is not None else None,
                "exit_code": exit_code,
                "probe_path": replica.probe_path,
                "last_error": replica.last_error,
            }
        )

    backend_pid = details[0]["pid"] if details else None
    backend_exit_code = next((detail["exit_code"] for detail in details if detail["exit_code"] is not None), None)
    backend_process_alive = any(detail["process_alive"] for detail in details)
    backend_ready_count = sum(1 for detail in details if detail["ready"])
    return backend_pid, backend_exit_code, backend_process_alive, backend_ready_count, details


async def get_health_payload() -> dict[str, Any]:
    healthy, backend_health, message = await _probe_backend_health()
    if healthy:
        global _backend_ready, _backend_last_error
        _backend_ready = True
        _backend_last_error = ""
    elif not _backend_alive() and _settings.manage_backend_process:
        _backend_ready = False
        if message:
            _backend_last_error = message

    backend_pid: Optional[int] = None
    backend_exit_code: Optional[int] = None
    backend_process_alive = False
    backend_ready_count = 1 if healthy else 0
    backend_replica_details: list[dict[str, Any]] = []
    with _backend_lock:
        if _settings.manage_backend_process:
            (
                backend_pid,
                backend_exit_code,
                backend_process_alive,
                backend_ready_count,
                backend_replica_details,
            ) = _managed_backend_runtime_locked()

    backend_state = _backend_state(healthy, backend_process_alive, backend_exit_code)
    backend_message = _backend_message(healthy, backend_process_alive, backend_exit_code, message)

    return {
        "status": "ok" if healthy else "degraded",
        "backend": "vllm",
        "backend_ready": healthy,
        "backend_ready_count": backend_ready_count,
        "backend_replica_count": len(backend_replica_details) if backend_replica_details else 1,
        "backend_state": backend_state,
        "backend_process_alive": backend_process_alive,
        "backend_url": _settings.backend_base_url,
        "backend_urls": [detail["base_url"] for detail in backend_replica_details] or [_settings.backend_base_url],
        "backend_probe_path": _backend_probe_path,
        "backend_pid": backend_pid,
        "backend_exit_code": backend_exit_code,
        "backend_last_error": backend_message,
        "backend_health": backend_health,
        "backend_replicas": backend_replica_details,
        "model_id": _settings.model_id,
        "model_revision": _settings.model_revision,
        "port": PORT,
        "backend_port": _settings.backend_port,
        "dtype": _settings.dtype,
        "backend_target_device": "cuda",
        "cpu_fallback": False,
        "max_model_len": _settings.max_model_len,
        "max_dimensions": _settings.max_dimensions,
        "gpu_memory_utilization": _settings.gpu_memory_utilization,
        "default_query_instruction": _settings.default_query_instruction,
        "manage_backend_process": _settings.manage_backend_process,
        "preload_model": _settings.preload_model,
        "auto_backend_replicas": _env_flag(_AUTO_BACKEND_REPLICAS_ENV, "1"),
        "started_at": _backend_started_at,
        "server_time": datetime.now().astimezone().isoformat(),
        "timezone": datetime.now().astimezone().tzname(),
    }


def get_health_snapshot() -> dict[str, Any]:
    backend_pid: Optional[int] = None
    backend_exit_code: Optional[int] = None
    backend_process_alive = False
    backend_ready_count = 1 if _backend_ready else 0
    backend_replica_details: list[dict[str, Any]] = []
    with _backend_lock:
        if _settings.manage_backend_process:
            (
                backend_pid,
                backend_exit_code,
                backend_process_alive,
                backend_ready_count,
                backend_replica_details,
            ) = _managed_backend_runtime_locked()

    backend_state = _backend_state(_backend_ready, backend_process_alive, backend_exit_code)
    backend_message = _backend_message(_backend_ready, backend_process_alive, backend_exit_code, "")

    return {
        "status": "ok" if _backend_ready else "degraded",
        "backend": "vllm",
        "backend_ready": _backend_ready,
        "backend_ready_count": backend_ready_count,
        "backend_replica_count": len(backend_replica_details) if backend_replica_details else 1,
        "backend_state": backend_state,
        "backend_process_alive": backend_process_alive,
        "backend_url": _settings.backend_base_url,
        "backend_urls": [detail["base_url"] for detail in backend_replica_details] or [_settings.backend_base_url],
        "backend_probe_path": _backend_probe_path,
        "backend_pid": backend_pid,
        "backend_exit_code": backend_exit_code,
        "backend_last_error": backend_message,
        "backend_replicas": backend_replica_details,
        "model_id": _settings.model_id,
        "model_revision": _settings.model_revision,
        "port": PORT,
        "backend_port": _settings.backend_port,
        "dtype": _settings.dtype,
        "backend_target_device": "cuda",
        "cpu_fallback": False,
        "max_model_len": _settings.max_model_len,
        "max_dimensions": _settings.max_dimensions,
        "gpu_memory_utilization": _settings.gpu_memory_utilization,
        "default_query_instruction": _settings.default_query_instruction,
        "manage_backend_process": _settings.manage_backend_process,
        "preload_model": _settings.preload_model,
        "auto_backend_replicas": _env_flag(_AUTO_BACKEND_REPLICAS_ENV, "1"),
        "started_at": _backend_started_at,
        "server_time": datetime.now().astimezone().isoformat(),
        "timezone": datetime.now().astimezone().tzname(),
    }


async def reload_backend(new_config: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "model_id",
        "model_revision",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "default_query_instruction",
        "extra_args",
    }
    unknown = set(new_config) - allowed_fields
    if unknown:
        raise InputValidationError(f"Unsupported reload fields: {', '.join(sorted(unknown))}")

    candidate_model_id = str(new_config.get("model_id") or _settings.model_id)
    candidate_model_revision = (
        new_config.get("model_revision") or None
        if "model_revision" in new_config
        else _settings.model_revision
    )
    candidate_max_dimensions = resolve_model_max_dimensions(
        candidate_model_id,
        candidate_model_revision,
        _settings.hf_home,
    )

    with _backend_lock:
        _stop_backend_process_locked()

        if "model_id" in new_config and new_config["model_id"]:
            _settings.model_id = str(new_config["model_id"])
        if "model_revision" in new_config:
            _settings.model_revision = new_config["model_revision"] or None
        _settings.max_dimensions = candidate_max_dimensions
        if "dtype" in new_config and new_config["dtype"]:
            _settings.dtype = str(new_config["dtype"])
        if "max_model_len" in new_config and new_config["max_model_len"] is not None:
            _settings.max_model_len = int(new_config["max_model_len"])
        if "gpu_memory_utilization" in new_config and new_config["gpu_memory_utilization"] is not None:
            _settings.gpu_memory_utilization = float(new_config["gpu_memory_utilization"])
        if "default_query_instruction" in new_config and new_config["default_query_instruction"]:
            _settings.default_query_instruction = str(new_config["default_query_instruction"])
        if "extra_args" in new_config:
            _settings.extra_args = str(new_config["extra_args"] or "")

        _settings.backend_base_url = VLLM_BASE_URL or f"http://{_settings.backend_host}:{_settings.backend_port}"
        _start_backend_process_locked()

    await wait_for_backend_ready()
    return await get_health_payload()
