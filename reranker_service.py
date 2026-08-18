import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from embedding_service import BackendProxyError, BackendUnavailableError, InputValidationError
from process_utils import terminate_process_group


RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B")
RERANKER_MODEL_REVISION = os.getenv("RERANKER_MODEL_REVISION")
RERANKER_BACKEND_HOST = os.getenv("RERANKER_BACKEND_HOST", "127.0.0.1")
RERANKER_BACKEND_PORT = int(os.getenv("RERANKER_BACKEND_PORT", "8002"))
RERANKER_DTYPE = os.getenv("RERANKER_DTYPE", "float16")
RERANKER_MAX_MODEL_LEN = int(os.getenv("RERANKER_MAX_MODEL_LEN", "2048"))
RERANKER_GPU_MEMORY_UTILIZATION = float(os.getenv("RERANKER_GPU_MEMORY_UTILIZATION", "0.08"))
RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION = float(
    os.getenv("RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION", "0.085")
)
RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION = float(
    os.getenv("RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION", "0.06")
)
RERANKER_MAX_NUM_SEQS = 1
RERANKER_MAX_NUM_BATCHED_TOKENS = 2048
RERANKER_HTTP_TIMEOUT = float(os.getenv("RERANKER_HTTP_TIMEOUT", "120"))
RERANKER_START_TIMEOUT = int(os.getenv("RERANKER_START_TIMEOUT", "600"))
RERANKER_POLL_INTERVAL = float(os.getenv("RERANKER_POLL_INTERVAL", "1"))
RERANKER_REQUEST_READY_TIMEOUT = float(os.getenv("RERANKER_REQUEST_READY_TIMEOUT", "3"))
RERANKER_PRELOAD = os.getenv("RERANKER_PRELOAD", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
RERANKER_MANAGE_PROCESS = os.getenv("RERANKER_MANAGE_PROCESS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
RERANKER_QUANTIZATION = os.getenv("RERANKER_QUANTIZATION", "none").strip().lower()
RERANKER_AUTO_FALLBACK_4BIT = os.getenv("RERANKER_AUTO_FALLBACK_4BIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
RERANKER_BASE_URL = os.getenv(
    "RERANKER_BASE_URL",
    f"http://{RERANKER_BACKEND_HOST}:{RERANKER_BACKEND_PORT}",
).rstrip("/")
RERANKER_TEMPLATE_PATH = os.getenv(
    "RERANKER_TEMPLATE_PATH",
    str(Path(__file__).resolve().parent / "templates" / "qwen3_reranker.jinja"),
)
RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANKER_MAX_DOCUMENTS = 50
RERANKER_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


@dataclass
class RerankerSettings:
    model_id: str = RERANKER_MODEL_ID
    model_revision: Optional[str] = RERANKER_MODEL_REVISION
    backend_host: str = RERANKER_BACKEND_HOST
    backend_port: int = RERANKER_BACKEND_PORT
    backend_base_url: str = RERANKER_BASE_URL
    dtype: str = RERANKER_DTYPE
    max_model_len: int = RERANKER_MAX_MODEL_LEN
    max_num_seqs: int = RERANKER_MAX_NUM_SEQS
    max_num_batched_tokens: int = RERANKER_MAX_NUM_BATCHED_TOKENS
    gpu_memory_utilization: float = (
        RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION
        if RERANKER_QUANTIZATION == "bitsandbytes"
        else RERANKER_GPU_MEMORY_UTILIZATION
    )
    quantization: Literal["none", "bitsandbytes"] = "bitsandbytes" if RERANKER_QUANTIZATION == "bitsandbytes" else "none"
    hf_home: str = os.getenv("HF_HOME", "/models")
    template_path: str = RERANKER_TEMPLATE_PATH
    instruction: str = RERANKER_INSTRUCTION
    manage_process: bool = RERANKER_MANAGE_PROCESS
    preload: bool = RERANKER_PRELOAD


_settings = RerankerSettings()
_lock = threading.RLock()
_process: Optional[subprocess.Popen[Any]] = None
_ready = False
_started_at: Optional[float] = None
_last_error = ""
_probe_path = ""
_reload_in_progress = False
_fallback_attempts: list[str] = []


def _backend_env() -> dict[str, str]:
    env = dict(os.environ)
    env["HF_HOME"] = _settings.hf_home
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    env.pop("VLLM_PORT", None)
    no_proxy = (env.get("NO_PROXY") or "localhost,127.0.0.1").strip()
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _build_vllm_command() -> list[str]:
    command = [
        "vllm",
        "serve",
        _settings.model_id,
        "--host",
        _settings.backend_host,
        "--port",
        str(_settings.backend_port),
        "--runner",
        "pooling",
        "--convert",
        "classify",
        "--dtype",
        _settings.dtype,
        "--max-model-len",
        str(_settings.max_model_len),
        "--max-num-seqs",
        str(_settings.max_num_seqs),
        "--max-num-batched-tokens",
        str(_settings.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(_settings.gpu_memory_utilization),
        "--served-model-name",
        _settings.model_id,
        "--hf-overrides",
        json.dumps(RERANKER_HF_OVERRIDES, separators=(",", ":")),
        "--chat-template",
        _settings.template_path,
        "--enforce-eager",
    ]
    if _settings.model_revision:
        command.extend(["--revision", _settings.model_revision])
    if _settings.quantization == "bitsandbytes":
        command.extend(["--quantization", "bitsandbytes"])
    return command


def _process_alive_locked() -> bool:
    return _process is not None and _process.poll() is None


def _state_locked() -> str:
    if _reload_in_progress:
        return "reloading"
    if _ready:
        return "ready"
    if _process_alive_locked():
        return "starting"
    if _process is not None and _process.poll() is not None:
        return "exited"
    return "stopped"


def _start_locked() -> None:
    global _process, _ready, _started_at, _last_error
    if not _settings.manage_process or _process_alive_locked():
        return
    if not Path(_settings.template_path).is_file():
        raise BackendUnavailableError(f"Reranker template not found: {_settings.template_path}")
    try:
        _process = subprocess.Popen(
            _build_vllm_command(),
            env=_backend_env(),
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        _process = None
        _last_error = "Failed to start Reranker: `vllm` command not found."
        raise BackendUnavailableError(_last_error) from exc
    _ready = False
    _last_error = ""
    _started_at = time.time()


def _stop_locked() -> None:
    global _process, _ready
    process = _process
    _process = None
    _ready = False
    if process is None:
        return
    terminate_process_group(process)


async def _probe() -> tuple[bool, Optional[dict[str, Any]], str]:
    global _probe_path
    timeout = httpx.Timeout(3, connect=1)
    last_message = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ("/health", "/v1/models"):
            try:
                response = await client.get(f"{_settings.backend_base_url}{path}")
            except httpx.HTTPError as exc:
                last_message = str(exc)
                continue
            if response.status_code >= 400:
                last_message = f"{path} returned HTTP {response.status_code}"
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = None
            _probe_path = path
            return True, payload, ""
    _probe_path = ""
    return False, None, last_message or "Reranker backend is not reachable."


async def wait_for_reranker_ready(timeout_s: Optional[float] = None) -> None:
    global _ready, _last_error
    deadline = time.monotonic() + float(timeout_s or RERANKER_START_TIMEOUT)
    while time.monotonic() < deadline:
        with _lock:
            alive = _process_alive_locked()
            exit_code = _process.poll() if _process is not None else None
        if _settings.manage_process and not alive:
            _ready = False
            _last_error = (
                f"Reranker vLLM exited with code {exit_code}."
                if exit_code is not None
                else "Reranker process is not running."
            )
            raise BackendUnavailableError(_last_error)
        healthy, _, message = await _probe()
        if healthy:
            _ready = True
            _last_error = ""
            return
        _ready = False
        _last_error = message
        await asyncio.sleep(RERANKER_POLL_INTERVAL)
    _ready = False
    _last_error = f"Timed out after {int(timeout_s or RERANKER_START_TIMEOUT)}s waiting for Reranker."
    raise BackendUnavailableError(_last_error)


async def ensure_reranker_started(
    wait_ready: bool = True,
    timeout_s: Optional[float] = None,
) -> None:
    with _lock:
        _start_locked()
    if wait_ready:
        await wait_for_reranker_ready(timeout_s)


async def maybe_preload_reranker() -> None:
    global _last_error
    if not _settings.preload:
        return
    attempts: list[tuple[float, Literal["none", "bitsandbytes"]]] = [
        (_settings.gpu_memory_utilization, _settings.quantization)
    ]
    if _settings.quantization == "none" and _settings.gpu_memory_utilization < RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION:
        attempts.append((RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION, "none"))
    if _settings.quantization == "none" and RERANKER_AUTO_FALLBACK_4BIT:
        attempts.append((RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION, "bitsandbytes"))

    _fallback_attempts.clear()
    for gpu_utilization, quantization in attempts:
        _settings.gpu_memory_utilization = gpu_utilization
        _settings.quantization = quantization
        _fallback_attempts.append(
            f"gpu_memory_utilization={gpu_utilization},quantization={quantization}"
        )
        try:
            await ensure_reranker_started(wait_ready=True)
            return
        except Exception as exc:
            _last_error = str(exc)
            with _lock:
                _stop_locked()


async def shutdown_reranker() -> None:
    with _lock:
        _stop_locked()


def get_current_reranker_model_id() -> str:
    return _settings.model_id


def get_reranker_settings() -> dict[str, Any]:
    return asdict(_settings)


def prepare_rerank_payload(request_payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    query = request_payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise InputValidationError("`query` must be a non-empty string.")

    raw_documents = request_payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise InputValidationError("`documents` must contain between 1 and 50 strings.")
    if len(raw_documents) > RERANKER_MAX_DOCUMENTS:
        raise InputValidationError("`documents` must contain at most 50 items.")
    documents: list[str] = []
    for index, document in enumerate(raw_documents):
        if not isinstance(document, str):
            raise InputValidationError(f"`documents[{index}]` must be a string.")
        if not document.strip():
            raise InputValidationError(f"`documents[{index}]` must not be empty.")
        documents.append(document)

    requested_model = request_payload.get("model")
    if requested_model is not None and requested_model != _settings.model_id:
        raise InputValidationError(
            f"Requested model {requested_model!r} does not match the served model {_settings.model_id!r}."
        )

    top_n = request_payload.get("top_n")
    if top_n is None:
        top_n = len(documents)
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise InputValidationError("`top_n` must be an integer.")
    if top_n < 1 or top_n > len(documents):
        raise InputValidationError(f"`top_n` must be between 1 and {len(documents)}.")

    payload: dict[str, Any] = {
        "model": _settings.model_id,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    if request_payload.get("user") is not None:
        payload["user"] = request_payload["user"]
    return payload, documents


async def _post_rerank(payload: dict[str, Any]) -> dict[str, Any]:
    global _ready, _last_error
    timeout = httpx.Timeout(RERANKER_HTTP_TIMEOUT, connect=5)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{_settings.backend_base_url}/v1/rerank", json=payload)
    except httpx.HTTPError as exc:
        _ready = False
        _last_error = str(exc)
        raise BackendUnavailableError(f"Failed to reach Reranker backend: {exc}") from exc
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
            f"Reranker backend returned HTTP {response.status_code}",
            status_code=response.status_code,
            payload=error_payload,
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise BackendProxyError("Reranker backend returned non-JSON response.", 502) from exc
    _ready = True
    _last_error = ""
    return result


def _normalize_response(
    response_payload: dict[str, Any],
    documents: list[str],
    top_n: int,
) -> dict[str, Any]:
    results = response_payload.get("results")
    if not isinstance(results, list):
        raise BackendProxyError("Reranker backend response is missing `results`.", 502)
    normalized_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            raise BackendProxyError("Reranker backend returned an invalid result item.", 502)
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(documents):
            raise BackendProxyError("Reranker backend returned an invalid document index.", 502)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise BackendProxyError("Reranker backend returned an invalid relevance score.", 502)
        normalized_results.append(
            {
                "index": index,
                "document": {"text": documents[index]},
                "relevance_score": float(score),
            }
        )
    normalized_results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {
        "id": response_payload.get("id") or f"rerank-{uuid.uuid4().hex}",
        "model": response_payload.get("model") or _settings.model_id,
        "usage": response_payload.get("usage") or {"prompt_tokens": 0, "total_tokens": 0},
        "results": normalized_results[:top_n],
    }


async def create_rerank(request_payload: dict[str, Any]) -> dict[str, Any]:
    backend_payload, documents = prepare_rerank_payload(request_payload)
    try:
        await ensure_reranker_started(True, RERANKER_REQUEST_READY_TIMEOUT)
    except BackendUnavailableError as exc:
        with _lock:
            alive = _process_alive_locked()
        if alive:
            raise BackendUnavailableError(
                "Reranker backend is still starting. Check /health and retry when reranker.ready is true."
            ) from exc
        raise
    response_payload = await _post_rerank(backend_payload)
    return _normalize_response(response_payload, documents, backend_payload["top_n"])


async def rerank_documents(
    query: str,
    documents: list[str],
    top_n: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    return await create_rerank(payload)


def get_reranker_health_snapshot() -> dict[str, Any]:
    with _lock:
        process_alive = _process_alive_locked()
        pid = _process.pid if _process is not None else None
        exit_code = _process.poll() if _process is not None else None
        state = _state_locked()
    return {
        "ready": _ready,
        "state": state,
        "process_alive": process_alive,
        "pid": pid,
        "exit_code": exit_code,
        "url": _settings.backend_base_url,
        "port": _settings.backend_port,
        "probe_path": _probe_path,
        "model": _settings.model_id,
        "model_revision": _settings.model_revision,
        "dtype": _settings.dtype,
        "max_model_len": _settings.max_model_len,
        "max_num_seqs": _settings.max_num_seqs,
        "max_num_batched_tokens": _settings.max_num_batched_tokens,
        "gpu_memory_utilization": _settings.gpu_memory_utilization,
        "quantization": _settings.quantization,
        "instruction": _settings.instruction,
        "last_error": _last_error,
        "started_at": _started_at,
        "preload": _settings.preload,
        "fallback_attempts": list(_fallback_attempts),
    }


async def get_reranker_health_payload() -> dict[str, Any]:
    global _ready, _last_error
    healthy, _, message = await _probe()
    if healthy:
        _ready = True
        _last_error = ""
    else:
        with _lock:
            alive = _process_alive_locked()
        _ready = False
        if not alive or not _settings.manage_process:
            _last_error = message
    payload = get_reranker_health_snapshot()
    payload["server_time"] = datetime.now().astimezone().isoformat()
    return payload


async def reload_reranker(new_config: dict[str, Any]) -> dict[str, Any]:
    global _reload_in_progress, _last_error
    unknown = set(new_config) - {"model_revision", "quantization"}
    if unknown:
        raise InputValidationError(f"Unsupported Reranker reload fields: {', '.join(sorted(unknown))}")
    quantization = new_config.get("quantization", _settings.quantization)
    if quantization not in ("none", "bitsandbytes"):
        raise InputValidationError("`quantization` must be `none` or `bitsandbytes`.")

    _reload_in_progress = True
    try:
        with _lock:
            _stop_locked()
            if "model_revision" in new_config:
                _settings.model_revision = new_config["model_revision"] or None
            _settings.quantization = quantization
            _settings.gpu_memory_utilization = (
                RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION
                if quantization == "bitsandbytes"
                else RERANKER_GPU_MEMORY_UTILIZATION
            )
            _start_locked()
        await wait_for_reranker_ready()
    except Exception as exc:
        _last_error = str(exc)
        raise
    finally:
        _reload_in_progress = False
    return await get_reranker_health_payload()
