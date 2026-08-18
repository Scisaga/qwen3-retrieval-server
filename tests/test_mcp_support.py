import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app
import mcp_server


def test_usage_resource_includes_projector_tool():
    content = mcp_server.build_usage_resource_content()
    assert "embed_text" in content
    assert "project_texts" in content
    assert "projection_method" in content
    assert "rerank_documents" in content


def test_health_resource_content_is_json():
    content = mcp_server.build_health_resource_content()
    payload = json.loads(content)
    assert isinstance(payload, dict)
    assert "model_id" in payload
    assert "backend_ready" in payload
    assert "reranker" in payload


@pytest.mark.anyio
async def test_embed_text_impl_success(monkeypatch):
    mocked = AsyncMock(
        return_value={
            "object": "list",
            "model": "Qwen/Qwen3-Embedding-4B",
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }
    )
    monkeypatch.setattr(mcp_server, "embed_texts", mocked)

    payload = await mcp_server.embed_text_impl(
        texts=["hello"],
        input_type="query",
        instruction="Given a query, retrieve relevant passages",
        dimensions=128,
    )

    assert payload["object"] == "list"
    mocked.assert_awaited_once()


@pytest.mark.anyio
async def test_project_texts_impl_accepts_single_string(monkeypatch):
    mocked = AsyncMock(
        return_value={
            "object": "projector",
            "points": [],
            "neighbors": {},
            "projection_meta": {"projection_method": "umap"},
        }
    )
    monkeypatch.setattr(mcp_server, "create_projector_payload", mocked)

    payload = await mcp_server.project_texts_impl(
        texts="hello",
        labels=None,
        projection_method="umap",
        metric="cosine",
        neighbors_k=8,
        point_size=4.0,
    )

    assert payload["object"] == "projector"
    call_payload = mocked.await_args.args[0]
    assert call_payload["inputs"] == ["hello"]
    assert call_payload["neighbors_k"] == 8


@pytest.mark.anyio
async def test_rerank_documents_impl(monkeypatch):
    mocked = AsyncMock(return_value={"id": "r1", "results": [{"index": 0, "relevance_score": 0.8}]})
    monkeypatch.setattr(mcp_server, "rerank_documents_service", mocked)

    payload = await mcp_server.rerank_documents_impl("query", ["document"], top_n=1)

    assert payload["id"] == "r1"
    mocked.assert_awaited_once_with(query="query", documents=["document"], top_n=1)


def test_fastapi_mcp_mount(monkeypatch):
    async def _noop():
        return None

    monkeypatch.setattr(app, "maybe_preload_backend", _noop)
    monkeypatch.setattr(app, "maybe_preload_reranker", _noop)
    monkeypatch.setattr(app, "shutdown_backend", _noop)
    monkeypatch.setattr(app, "shutdown_reranker", _noop)

    with TestClient(app.create_application()) as client:
        response = client.get("/mcp", follow_redirects=False)

    assert response.status_code != 404
