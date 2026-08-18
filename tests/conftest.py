import pytest

import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _disable_real_model_lifecycle(monkeypatch):
    async def noop() -> None:
        return None

    monkeypatch.setattr(app, "maybe_preload_backend", noop)
    monkeypatch.setattr(app, "maybe_preload_reranker", noop)
    monkeypatch.setattr(app, "shutdown_reranker", noop)
    monkeypatch.setattr(app, "shutdown_backend", noop)
