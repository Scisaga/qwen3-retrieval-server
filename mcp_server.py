import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from projector_service import ProjectorDependencyError, create_projector_payload
from embedding_service import (
    BackendUnavailableError,
    BackendProxyError,
    DEFAULT_QUERY_INSTRUCTION,
    InputValidationError,
    embed_texts,
)
from reranker_service import rerank_documents as rerank_documents_service
from service_health import get_aggregate_health_snapshot

def build_health_resource_content() -> str:
    return json.dumps(get_aggregate_health_snapshot(), ensure_ascii=False, indent=2)


def build_usage_resource_content() -> str:
    lines = [
        "# Qwen3 Embedding & Reranker MCP Usage",
        "",
        "Use `embed_text` for retrieval embeddings, `rerank_documents` for second-stage ranking, and `project_texts` for projector payloads.",
        "",
        "Tool: embed_text",
        "- `texts` (required): one string or a list of strings",
        "- `input_type` (optional): `query` or `document`",
        "- `instruction` (optional): query-side instruction for Qwen retrieval formatting",
        "- `dimensions` (optional): output dimensions between 32 and the current model's `max_dimensions` from health",
        "",
        "Tool: project_texts",
        "- `texts` (required): one string or a list of strings",
        "- `labels` (optional): same length as `texts`",
        "- `projection_method` (optional): `umap`, `tsne`, or `pca`",
        "- `metric` (optional): `cosine` or `euclidean`",
        "- `neighbors_k` (optional): 1-256",
        "- `point_size` (optional): (0, 64]",
        "- `input_type` / `instruction` are also supported for retrieval-style query/document encoding",
        "",
        "Tool: rerank_documents",
        "- `query` (required): one non-empty query string",
        "- `documents` (required): 1-50 non-empty strings",
        "- `top_n` (optional): return the best 1..len(documents) results; omit to return all",
        "- The instruction and Qwen3 score template are fixed at service level.",
        "",
        "Notes:",
        "- Query embeddings are wrapped as `Instruct: ...\\nQuery:...` before being sent to Qwen.",
        f"- If `instruction` is omitted for queries, the service uses the default: `{DEFAULT_QUERY_INSTRUCTION}`",
        "- `project_texts` returns points, neighbors, and projection metadata; clients can render directly.",
    ]
    return "\n".join(lines)


async def embed_text_impl(
    texts: str | list[str],
    input_type: Optional[str] = None,
    instruction: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> dict:
    try:
        return await embed_texts(
            texts=texts,
            input_type=input_type,
            instruction=instruction,
            dimensions=dimensions,
        )
    except InputValidationError as exc:
        raise RuntimeError(f"invalid_input: {exc}") from exc
    except BackendUnavailableError as exc:
        raise RuntimeError(f"backend_unavailable: {exc}") from exc
    except BackendProxyError as exc:
        if exc.payload is not None:
            raise RuntimeError(json.dumps(exc.payload, ensure_ascii=False)) from exc
        raise RuntimeError(f"backend_error: {exc}") from exc


async def project_texts_impl(
    texts: str | list[str],
    labels: Optional[list[str]] = None,
    input_type: Optional[str] = None,
    instruction: Optional[str] = None,
    projection_method: str = "umap",
    metric: str = "cosine",
    neighbors_k: int = 10,
    point_size: float = 3.5,
) -> dict:
    normalized_inputs = [texts] if isinstance(texts, str) else texts
    payload: dict = {
        "inputs": normalized_inputs,
        "projection_method": projection_method,
        "metric": metric,
        "neighbors_k": neighbors_k,
        "point_size": point_size,
    }
    if labels is not None:
        payload["labels"] = labels
    if input_type is not None:
        payload["input_type"] = input_type
    if instruction is not None:
        payload["instruction"] = instruction

    try:
        return await create_projector_payload(payload)
    except InputValidationError as exc:
        raise RuntimeError(f"invalid_input: {exc}") from exc
    except ProjectorDependencyError as exc:
        raise RuntimeError(f"projector_dependency_error: {exc}") from exc
    except BackendUnavailableError as exc:
        raise RuntimeError(f"backend_unavailable: {exc}") from exc
    except BackendProxyError as exc:
        if exc.payload is not None:
            raise RuntimeError(json.dumps(exc.payload, ensure_ascii=False)) from exc
        raise RuntimeError(f"backend_error: {exc}") from exc


async def rerank_documents_impl(
    query: str,
    documents: list[str],
    top_n: Optional[int] = None,
) -> dict:
    try:
        return await rerank_documents_service(query=query, documents=documents, top_n=top_n)
    except InputValidationError as exc:
        raise RuntimeError(f"invalid_input: {exc}") from exc
    except BackendUnavailableError as exc:
        raise RuntimeError(f"backend_unavailable: {exc}") from exc
    except BackendProxyError as exc:
        if exc.payload is not None:
            raise RuntimeError(json.dumps(exc.payload, ensure_ascii=False)) from exc
        raise RuntimeError(f"backend_error: {exc}") from exc


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("Qwen3 Embedding & Reranker", stateless_http=True, json_response=True)
    mcp.settings.streamable_http_path = "/"

    @mcp.tool()
    async def embed_text(
        texts: str | list[str],
        input_type: Optional[str] = None,
        instruction: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> dict:
        """Generate text embeddings through the local vLLM-backed service."""
        return await embed_text_impl(
            texts=texts,
            input_type=input_type,
            instruction=instruction,
            dimensions=dimensions,
        )

    @mcp.tool()
    async def project_texts(
        texts: str | list[str],
        labels: Optional[list[str]] = None,
        input_type: Optional[str] = None,
        instruction: Optional[str] = None,
        projection_method: str = "umap",
        metric: str = "cosine",
        neighbors_k: int = 10,
        point_size: float = 3.5,
    ) -> dict:
        """Generate 3D projector payload with nearest neighbors."""
        return await project_texts_impl(
            texts=texts,
            labels=labels,
            input_type=input_type,
            instruction=instruction,
            projection_method=projection_method,
            metric=metric,
            neighbors_k=neighbors_k,
            point_size=point_size,
        )

    @mcp.tool()
    async def rerank_documents(
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> dict:
        """Rerank 1-50 candidate documents with the local Qwen3 Reranker."""
        return await rerank_documents_impl(query=query, documents=documents, top_n=top_n)

    @mcp.resource("qwen3embedding://health")
    def qwen3embedding_health() -> str:
        """Expose read-only runtime status for MCP clients."""
        return build_health_resource_content()

    @mcp.resource("qwen3embedding://usage")
    def qwen3embedding_usage() -> str:
        """Describe how to call the embedding tool safely."""
        return build_usage_resource_content()

    @mcp.prompt()
    def retrieval_embedding_workflow() -> str:
        """Guide retrieval-oriented clients to use the tool correctly."""
        return (
            "When embedding search queries, call `embed_text` with `input_type=query`. "
            "Pass a task-specific English `instruction` whenever possible. "
            "When embedding passages or chunks for indexing, use `input_type=document` and do not add the instruction. "
            "Keep query and document embeddings in separate requests."
        )

    @mcp.prompt()
    def rag_retrieval_workflow() -> str:
        """Guide clients through two-stage RAG retrieval."""
        return (
            "For two-stage RAG retrieval, first call `embed_text` with `input_type=query` and use the resulting "
            "vector to retrieve a broad candidate set from the client's vector store. Then pass the original query "
            "and up to 50 candidate passage strings to `rerank_documents`; use `top_n` to keep only the passages "
            "sent to generation. Preserve each result's original `index` when mapping it back to candidate metadata."
        )

    @mcp.prompt()
    def projector_workflow() -> str:
        """Guide clients to build projector-compatible payloads."""
        return (
            "When the client needs visual clustering or nearest-neighbor exploration, call `project_texts`. "
            "Use `labels` to encode categories shown in the scatter plot. "
            "Prefer `projection_method=umap` and `metric=cosine` for semantic text embeddings. "
            "Use `neighbors_k` between 5 and 20 for interactive browsing."
        )

    return mcp


mcp = create_mcp_server()
