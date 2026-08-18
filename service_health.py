from typing import Any

from embedding_service import get_health_payload as get_embedding_health_payload
from embedding_service import get_health_snapshot as get_embedding_health_snapshot
from reranker_service import get_reranker_health_payload, get_reranker_health_snapshot


def _aggregate(embedding: dict[str, Any], reranker: dict[str, Any]) -> dict[str, Any]:
    payload = dict(embedding)
    payload["reranker"] = reranker
    payload["status"] = "ok" if embedding.get("backend_ready") and reranker.get("ready") else "degraded"
    return payload


async def get_aggregate_health_payload() -> dict[str, Any]:
    embedding = await get_embedding_health_payload()
    reranker = await get_reranker_health_payload()
    return _aggregate(embedding, reranker)


def get_aggregate_health_snapshot() -> dict[str, Any]:
    return _aggregate(get_embedding_health_snapshot(), get_reranker_health_snapshot())
