from fastapi.testclient import TestClient
import pytest

import app
import embedding_service


@pytest.mark.anyio
async def test_preload_backends_is_sequential(monkeypatch):
    order = []

    async def embedding_preload():
        order.append("embedding")

    async def reranker_preload():
        order.append("reranker")

    monkeypatch.setattr(app, "maybe_preload_backend", embedding_preload)
    monkeypatch.setattr(app, "maybe_preload_reranker", reranker_preload)

    await app._preload_backends()

    assert order == ["embedding", "reranker"]


def test_shutdown_order_is_reranker_then_embedding(monkeypatch):
    order = []

    async def reranker_shutdown():
        order.append("reranker")

    async def embedding_shutdown():
        order.append("embedding")

    monkeypatch.setattr(app, "shutdown_reranker", reranker_shutdown)
    monkeypatch.setattr(app, "shutdown_backend", embedding_shutdown)

    with TestClient(app.create_application()):
        pass

    assert order == ["reranker", "embedding"]


@pytest.mark.anyio
async def test_health_status_is_degraded_until_both_backends_are_ready(monkeypatch):
    async def embedding_health():
        return {"status": "ok", "backend_ready": True, "model_id": "embedding"}

    async def reranker_health():
        return {"ready": False, "state": "starting"}

    monkeypatch.setattr(app, "get_embedding_health_payload", embedding_health)
    monkeypatch.setattr(app, "get_reranker_health_payload", reranker_health)

    payload = await app.get_health_payload()

    assert payload["backend_ready"] is True
    assert payload["status"] == "degraded"
    assert payload["reranker"]["state"] == "starting"


def test_index_html_contains_unified_navigation_and_projector_mount():
    html = app._build_index_html()

    assert 'data-tab="projector-section"' in html
    assert 'id="projector-root"' in html
    assert 'href="/projector-static/projector.css"' in html
    assert 'import("/projector-static/projector.js")' in html
    assert 'id="activeViewLabel"' in html
    assert 'id="similarityHeatmap"' in html
    assert 'class="payload-details"' in html
    assert 'id="topTimeChip"' not in html
    assert 'id="dimensions" type="number" min="32" max=' not in html
    assert "上限随模型" in html
    assert 'data-tab="reranker-section"' in html
    assert 'id="rerankQuery"' in html
    assert 'id="rerankerReloadBtn"' in html


def test_projector_page_redirects_to_embedded_tab():
    with TestClient(app.create_application(), follow_redirects=False) as client:
        response = client.get("/projector")

    assert response.status_code == 307
    assert response.headers["location"] == "/#projector-section"


def test_embeddings_route_returns_openai_shape(monkeypatch):
    async def fake_create_embeddings(payload):
        assert payload["input_type"] == "query"
        return {
            "object": "list",
            "model": "Qwen/Qwen3-Embedding-4B",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    monkeypatch.setattr(app, "create_embeddings", fake_create_embeddings)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "What is the capital of China?",
                "input_type": "query",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_health_route_reflects_backend_status(monkeypatch):
    async def fake_health():
        return {
            "status": "ok",
            "backend_ready": True,
            "model_id": "Qwen/Qwen3-Embedding-4B",
            "max_dimensions": 2560,
        }

    monkeypatch.setattr(app, "get_health_payload", fake_health)

    with TestClient(app.create_application()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["backend_ready"] is True


def test_rerank_route_returns_sorted_vllm_shape(monkeypatch):
    async def fake_create_rerank(payload):
        assert payload["top_n"] == 1
        return {
            "id": "rerank-1",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
            "results": [
                {
                    "index": 1,
                    "document": {"text": "北京是中国首都。"},
                    "relevance_score": 0.99,
                }
            ],
        }

    monkeypatch.setattr(app, "create_rerank", fake_create_rerank)
    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/rerank",
            json={"query": "中国首都是哪里？", "documents": ["巴黎", "北京是中国首都。"], "top_n": 1},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["index"] == 1


def test_rerank_route_rejects_extra_fields_with_unified_400():
    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["d"], "instruction": "not allowed"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_rerank_route_rejects_more_than_50_documents():
    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": [str(index) for index in range(51)]},
        )

    assert response.status_code == 400


def test_reranker_backend_error_is_passthrough(monkeypatch):
    async def fake_create_rerank(payload):
        raise embedding_service.BackendProxyError(
            "rate limited",
            status_code=429,
            payload={"error": {"message": "rate limited", "code": 429}},
        )

    monkeypatch.setattr(app, "create_rerank", fake_create_rerank)
    with TestClient(app.create_application()) as client:
        response = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "rate limited"


def test_admin_reload_requires_token(monkeypatch):
    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")

    with TestClient(app.create_application()) as client:
        response = client.post("/admin/reload", json={})

    assert response.status_code == 401


def test_admin_reload_success(monkeypatch):
    async def fake_reload(payload):
        assert payload["model_id"] == "Qwen/Qwen3-Embedding-4B"
        return {"status": "ok", "backend_ready": True}

    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(app, "reload_backend", fake_reload)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/admin/reload",
            headers={"x-admin-token": "secret"},
            json={"model_id": "Qwen/Qwen3-Embedding-4B"},
        )

    assert response.status_code == 200
    assert response.json()["backend_ready"] is True


def test_admin_reranker_reload_is_independent(monkeypatch):
    async def fake_reload(payload):
        assert payload == {"quantization": "none"}
        return {"ready": True, "quantization": "none"}

    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(app, "reload_reranker", fake_reload)
    with TestClient(app.create_application()) as client:
        response = client.post(
            "/admin/reranker/reload",
            headers={"x-admin-token": "secret"},
            json={"quantization": "none"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_admin_reranker_reload_rejects_runtime_limit_changes(monkeypatch):
    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")
    with TestClient(app.create_application()) as client:
        response = client.post(
            "/admin/reranker/reload",
            headers={"x-admin-token": "secret"},
            json={"gpu_memory_utilization": 0.2},
        )

    assert response.status_code == 400


def test_backend_proxy_error_is_passthrough(monkeypatch):
    async def fake_create_embeddings(payload):
        raise embedding_service.BackendProxyError(
            "bad request",
            status_code=400,
            payload={
                "error": {
                    "message": "dimensions not supported",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
        )

    monkeypatch.setattr(app, "create_embeddings", fake_create_embeddings)

    with TestClient(app.create_application()) as client:
        response = client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "dimensions not supported"


def test_projector_route_returns_payload(monkeypatch):
    async def fake_projector(payload, embedder):
        assert payload["projection_method"] == "pca"
        return {
            "object": "projector",
            "model": "Qwen/Qwen3-Embedding-4B",
            "points": [{"id": "0", "index": 0, "text": "hello", "label": "", "x": 0.0, "y": 0.0}],
            "neighbors": {"0": []},
            "projection_meta": {"projection_method": "pca", "cache_hit": False},
        }

    monkeypatch.setattr(app, "create_projector_payload", fake_projector)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/embeddings/projector",
            json={"inputs": ["hello", "world"], "projection_method": "pca"},
        )

    assert response.status_code == 200
    assert response.json()["object"] == "projector"


def test_projector_route_handles_input_validation_error(monkeypatch):
    async def fake_projector(payload, embedder):
        raise embedding_service.InputValidationError("bad projector request")

    monkeypatch.setattr(app, "create_projector_payload", fake_projector)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/embeddings/projector",
            json={"inputs": ["hello"], "projection_method": "pca"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad projector request"
