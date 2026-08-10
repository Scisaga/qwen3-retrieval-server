import pytest

import projector_service
from embedding_service import InputValidationError


def _clear_projector_cache():
    with projector_service._projector_cache_lock:
        projector_service._projector_cache.clear()


@pytest.mark.anyio
async def test_create_projector_payload_rejects_empty_inputs():
    _clear_projector_cache()
    with pytest.raises(InputValidationError):
        await projector_service.create_projector_payload(
            {"inputs": [], "projection_method": "pca"},
            embedder=None,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_create_projector_payload_rejects_mismatched_labels():
    _clear_projector_cache()
    with pytest.raises(InputValidationError):
        await projector_service.create_projector_payload(
            {
                "inputs": ["a", "b"],
                "labels": ["only-one"],
                "projection_method": "pca",
            },
            embedder=None,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_create_projector_payload_cache_hit():
    _clear_projector_cache()
    counter = {"calls": 0}

    async def fake_embedder(payload):
        counter["calls"] += 1
        return {
            "object": "list",
            "model": "Qwen/Qwen3-Embedding-4B",
            "data": [
                {"index": 0, "object": "embedding", "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "object": "embedding", "embedding": [0.2, 0.1, 0.4]},
            ],
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }

    request_payload = {
        "inputs": ["hello", "world"],
        "projection_method": "pca",
        "metric": "cosine",
        "neighbors_k": 1,
    }
    first = await projector_service.create_projector_payload(request_payload, embedder=fake_embedder)
    second = await projector_service.create_projector_payload(request_payload, embedder=fake_embedder)

    assert counter["calls"] == 1
    assert first["projection_meta"]["cache_hit"] is False
    assert second["projection_meta"]["cache_hit"] is True


@pytest.mark.anyio
async def test_create_projector_payload_contains_neighbors_and_points():
    _clear_projector_cache()

    async def fake_embedder(payload):
        return {
            "object": "list",
            "model": "Qwen/Qwen3-Embedding-4B",
            "data": [
                {"index": 0, "object": "embedding", "embedding": [1.0, 0.0, 0.0]},
                {"index": 1, "object": "embedding", "embedding": [0.9, 0.1, 0.0]},
                {"index": 2, "object": "embedding", "embedding": [0.0, 1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }

    payload = await projector_service.create_projector_payload(
        {
            "inputs": ["a", "b", "c"],
            "labels": ["L1", "L1", "L2"],
            "projection_method": "pca",
            "metric": "cosine",
            "neighbors_k": 2,
        },
        embedder=fake_embedder,
    )

    assert payload["object"] == "projector"
    assert len(payload["points"]) == 3
    assert payload["projection_meta"]["projection_method"] == "pca"
    assert payload["projection_meta"]["neighbors_k"] == 2
    assert len(payload["neighbors"]["0"]) == 2
