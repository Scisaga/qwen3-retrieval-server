#!/usr/bin/env python3
"""Measure the Reranker process tree's GPU memory at 100 ms intervals.

Run this script on the Docker host so NVML PIDs and /proc PIDs share a namespace.
It maps the container-visible Reranker PID from /health to its host PID, then
recursively includes every descendant process on GPU 0.
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def container_host_pid(container: str, namespace_pid: int) -> int:
    result = subprocess.run(
        ["docker", "top", container, "-eo", "pid"],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    for host_pid in candidates:
        try:
            status = Path(f"/proc/{host_pid}/status").read_text(encoding="utf-8")
        except OSError:
            continue
        nspid_line = next((line for line in status.splitlines() if line.startswith("NSpid:")), "")
        namespace_pids = [int(value) for value in nspid_line.split()[1:]]
        if namespace_pids and namespace_pids[-1] == namespace_pid:
            return host_pid
    raise RuntimeError(
        f"Could not map container PID {namespace_pid} to a host PID in container {container!r}."
    )


def descendants(root_pid: int) -> set[int]:
    parent_by_pid: dict[int, int] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
            pid = int(status_path.parent.name)
            ppid_line = next(line for line in lines if line.startswith("PPid:"))
            parent_by_pid[pid] = int(ppid_line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_by_pid.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def gpu_uuid(index: int) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def gpu_process_memory(target_pids: set[int], target_gpu_uuid: str) -> tuple[float, dict[int, float]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    memory_by_pid: dict[int, float] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            used_memory = float(fields[1])
        except ValueError:
            continue
        if pid in target_pids and fields[2] == target_gpu_uuid:
            memory_by_pid[pid] = used_memory
    return sum(memory_by_pid.values()), memory_by_pid


def run_concurrently(action: Callable[[], None], count: int) -> None:
    errors: list[BaseException] = []

    def guarded() -> None:
        try:
            action()
        except BaseException as exc:  # surfaced after every worker joins
            errors.append(exc)

    threads = [threading.Thread(target=guarded) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"{len(errors)} workload client(s) failed: {errors[0]}")


def sample_scenario(
    name: str,
    root_pid: int,
    target_gpu_uuid: str,
    action: Callable[[], None],
    interval: float,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    done = threading.Event()
    error: list[BaseException] = []

    def workload() -> None:
        try:
            action()
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=workload)
    worker.start()
    while not done.is_set() or not samples:
        pids = descendants(root_pid)
        total, details = gpu_process_memory(pids, target_gpu_uuid)
        samples.append({"at": time.time(), "total_mib": total, "processes": details})
        time.sleep(interval)
    worker.join()
    if error:
        raise error[0]
    peak = max(sample["total_mib"] for sample in samples)
    peak_sample = max(samples, key=lambda sample: sample["total_mib"])
    return {
        "scenario": name,
        "samples": len(samples),
        "peak_mib": peak,
        "peak_processes_mib": peak_sample["processes"],
    }


def tokenish_document(words: int, marker: str) -> str:
    return " ".join([marker, *(["context"] * max(0, words - 1))])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:12302")
    parser.add_argument("--container", default="qwen3_embedding_openai")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--limit-mib", type=float, default=2048)
    args = parser.parse_args()

    health = request_json(f"{args.base_url.rstrip('/')}/health")
    reranker = health.get("reranker") or {}
    namespace_pid = reranker.get("pid")
    if not reranker.get("ready") or not isinstance(namespace_pid, int):
        raise RuntimeError(f"Reranker is not ready: {json.dumps(reranker, ensure_ascii=False)}")
    root_pid = container_host_pid(args.container, namespace_pid)
    target_gpu_uuid = gpu_uuid(args.gpu_index)
    rerank_url = f"{args.base_url.rstrip('/')}/v1/rerank"

    def post(query: str, documents: list[str], top_n: int | None = None) -> None:
        payload: dict[str, Any] = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        request_json(rerank_url, payload)

    scenarios: list[tuple[str, Callable[[], None]]] = [
        ("idle", lambda: time.sleep(args.idle_seconds)),
        ("single_document", lambda: post("capital of China", ["Beijing is the capital of China."])),
        (
            "50_documents_approx_512_tokens",
            lambda: post(
                "Find the relevant passage",
                [tokenish_document(512, f"candidate-{index}") for index in range(50)],
                10,
            ),
        ),
        (
            "near_2048_token_input",
            lambda: post("long context retrieval", [tokenish_document(1900, "long-context")]),
        ),
        (
            "four_concurrent_http_clients",
            lambda: run_concurrently(
                lambda: post(
                    "Which passage answers the query?",
                    [tokenish_document(256, f"concurrent-{index}") for index in range(8)],
                    3,
                ),
                4,
            ),
        ),
    ]
    results = [
        sample_scenario(name, root_pid, target_gpu_uuid, action, args.interval)
        for name, action in scenarios
    ]
    peak = max(result["peak_mib"] for result in results)
    report = {
        "definition": "NVML used_memory sum for Reranker root PID and all descendants",
        "container_pid": namespace_pid,
        "host_root_pid": root_pid,
        "gpu_index": args.gpu_index,
        "gpu_uuid": target_gpu_uuid,
        "interval_ms": round(args.interval * 1000),
        "limit_mib_exclusive": args.limit_mib,
        "peak_mib": peak,
        "passed": peak < args.limit_mib,
        "scenarios": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
