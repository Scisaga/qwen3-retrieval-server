import contextlib
import json
import os
import asyncio
from typing import Literal, Optional

from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from embedding_service import (
    ADMIN_TOKEN,
    DEFAULT_QUERY_INSTRUCTION,
    BackendProxyError,
    BackendUnavailableError,
    InputValidationError,
    create_embeddings,
    get_current_model_id,
    get_health_payload as get_embedding_health_payload,
    maybe_preload_backend,
    reload_backend,
    shutdown_backend,
)
from mcp_server import create_mcp_server
from projector_service import ProjectorDependencyError, create_projector_payload
from reranker_service import (
    create_rerank,
    get_current_reranker_model_id,
    get_reranker_health_payload,
    maybe_preload_reranker,
    reload_reranker,
    shutdown_reranker,
)


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | list[str]
    model: Optional[str] = None
    dimensions: Optional[int] = Field(default=None, ge=32)
    encoding_format: Optional[str] = None
    user: Optional[str] = None
    input_type: Optional[Literal["query", "document"]] = None
    instruction: Optional[str] = None


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    dtype: Optional[str] = None
    max_model_len: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    default_query_instruction: Optional[str] = None
    extra_args: Optional[str] = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    documents: list[str] = Field(min_length=1, max_length=50)
    top_n: Optional[int] = Field(default=None, ge=1)
    model: Optional[str] = None
    user: Optional[str] = None


class RerankerReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_revision: Optional[str] = None
    quantization: Optional[Literal["none", "bitsandbytes"]] = None


class ProjectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str]
    labels: Optional[list[str]] = None
    model: Optional[str] = None
    input_type: Optional[Literal["query", "document"]] = None
    instruction: Optional[str] = None
    projection_method: Literal["umap", "tsne", "pca"] = "umap"
    metric: Literal["cosine", "euclidean"] = "cosine"
    neighbors_k: int = Field(default=10, ge=1, le=256)
    point_size: float = Field(default=3.5, gt=0.0, le=64.0)


def _error_response(message: str, status_code: int, error_type: str, code: Optional[int | str] = None) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(payload, status_code=status_code)


def _authorize_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise PermissionError("Invalid x-admin-token.")


async def get_health_payload() -> dict:
    embedding = await get_embedding_health_payload()
    reranker = await get_reranker_health_payload()
    payload = dict(embedding)
    payload["reranker"] = reranker
    payload["status"] = "ok" if payload.get("backend_ready") and reranker.get("ready") else "degraded"
    return payload


async def _preload_backends() -> None:
    await maybe_preload_backend()
    await maybe_preload_reranker()


def _build_index_html() -> str:
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="theme-color" content="#0a1224"/>
  <link rel="icon" type="image/png" href="/static/logo.png"/>
  <link rel="apple-touch-icon" href="/static/logo.png"/>
  <link rel="stylesheet" href="/projector-static/projector.css"/>
  <title>Qwen3 Embedding &amp; Reranker</title>
  <style>
    :root{
      --bg0:#050913;
      --bg1:#0a1224;
      --panel:rgba(10, 18, 36, .82);
      --panel2:rgba(13, 22, 42, .92);
      --border:rgba(148,163,184,.18);
      --text:rgba(226,232,240,.94);
      --muted:rgba(148,163,184,.86);
      --muted2:rgba(148,163,184,.62);
      --accent:#ff8a1f;
      --accent2:#ffd166;
      --good:#22c55e;
      --danger:#ef4444;
      --warn:#f59e0b;
      --shadow:0 24px 80px rgba(0,0,0,.42);
      --radius:10px;
      --radius2:12px;
    }
    *{box-sizing:border-box}
    html,body{min-height:100%;margin:0}
    body{
      color:var(--text);
      font-family:"IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
      overflow-x:hidden;
      background:
        radial-gradient(1100px 700px at 10% -10%, rgba(255,138,31,.18), transparent 60%),
        radial-gradient(900px 600px at 100% 0%, rgba(96,165,250,.16), transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }
    a{color:inherit;text-decoration:none}
    code{
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
      background:rgba(2,6,23,.52);
      border:1px solid var(--border);
      padding:2px 8px;
      border-radius:999px;
      font-size:.92em;
    }
    .app{
      display:grid;
      grid-template-columns:clamp(248px,22vw,312px) minmax(0,1fr);
      min-height:100vh;
      align-items:start;
      gap:12px;
      padding:12px;
    }
    .sidebar{
      width:auto;
      min-width:0;
      padding:16px 14px;
      border:1px solid var(--border);
      border-radius:12px;
      background:rgba(2,6,23,.54);
      backdrop-filter:blur(12px);
      position:sticky;
      top:12px;
      height:auto;
      max-height:calc(100vh - 24px);
      overflow-x:hidden;
      overflow-y:auto;
    }
    .brand{display:flex;gap:12px;align-items:center;padding:10px 8px 14px}
    .logo{width:40px;height:40px;border-radius:10px;display:block;object-fit:cover;box-shadow:0 12px 28px rgba(255,138,31,.26)}
    .brand-title{font-size:16px;font-weight:700;letter-spacing:.2px}
    .brand-sub{font-size:12px;color:var(--muted2);margin-top:2px}
    .nav{display:flex;flex-direction:column;gap:8px}
    .nav .nav-link{
      width:100%;
      appearance:none;
      border:none;
      text-align:left;
      display:flex;gap:10px;align-items:center;
      padding:10px 12px;border-radius:10px;border:1px solid transparent;
      background:transparent;
      color:var(--muted);transition:background .16s,border-color .16s,color .16s;
      cursor:pointer;
    }
    .nav .nav-link:hover{background:rgba(148,163,184,.08)}
    .nav .nav-link.active{
      background:rgba(255,138,31,.12);
      border-color:rgba(255,138,31,.22);
      color:var(--text);
    }
    .nav .dot{
      width:9px;height:9px;border-radius:999px;background:rgba(148,163,184,.54);
      box-shadow:0 0 0 4px rgba(148,163,184,.08)
    }
    .nav .nav-link.active .dot{
      background:var(--accent);
      box-shadow:0 0 0 4px rgba(255,138,31,.14)
    }
    .sidebar-card{
      margin-top:16px;padding:14px;border-radius:var(--radius);
      border:1px solid var(--border);background:rgba(15,23,42,.42)
    }
    .sidebar-card h3{margin:0 0 10px;font-size:13px}
    .sidebar-card p,.sidebar-card li{margin:0;color:var(--muted);font-size:12px;line-height:1.65;overflow-wrap:anywhere}
    .sidebar-card ul{padding-left:16px;margin:0}
    .service-endpoints{
      display:inline-flex;
      flex-wrap:wrap;
      gap:6px;
      max-width:100%;
      vertical-align:middle;
    }
    .main{min-width:0;display:flex;flex-direction:column}
    .topbar{
      position:sticky;top:0;z-index:12;
      display:flex;justify-content:space-between;align-items:center;gap:12px;
      padding:14px 18px;border-bottom:1px solid var(--border);
      background:rgba(2,6,23,.52);backdrop-filter:blur(12px)
    }
    .crumbs{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted)}
    .crumbs strong{color:var(--text)}
    .chips{display:flex;gap:10px;flex-wrap:wrap}
    .chip{
      display:inline-flex;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;
      border:1px solid var(--border);background:rgba(15,23,42,.5);font-size:12px;color:var(--muted)
    }
    .chip.ok{color:#bbf7d0;border-color:rgba(34,197,94,.26);background:rgba(34,197,94,.12)}
    .chip.warn{color:#fde68a;border-color:rgba(245,158,11,.28);background:rgba(245,158,11,.12)}
    .chip.bad{color:#fecaca;border-color:rgba(239,68,68,.26);background:rgba(239,68,68,.12)}
    .content{padding:18px 16px 32px}
    .tabbar{
      max-width:1320px;margin:0 auto 14px;display:flex;gap:8px;flex-wrap:wrap
    }
    .tab-btn{
      appearance:none;border:1px solid var(--border);background:rgba(15,23,42,.45);
      color:var(--muted);padding:8px 12px;border-radius:10px;cursor:pointer;font-size:13px;
      display:inline-flex;align-items:center
    }
    .tab-btn.active{
      color:var(--text);
      background:rgba(255,138,31,.12);
      border-color:rgba(255,138,31,.25);
    }
    .section{max-width:1320px;margin:0 auto 16px;display:none}
    .section.active{display:block}
    .section-head{margin:0 0 14px}
    .section-head h1,.section-head h2{margin:0;font-size:24px;letter-spacing:.2px}
    .section-head p{margin:8px 0 0;color:var(--muted);font-size:13px;line-height:1.7}
    .grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .card{
      border:1px solid var(--border);border-radius:var(--radius2);background:var(--panel);
      box-shadow:var(--shadow);overflow:hidden
    }
    .rerank-results{display:flex;flex-direction:column;gap:10px;margin-top:12px}
    .rerank-result{border:1px solid var(--border);border-radius:10px;padding:12px;background:rgba(2,6,23,.38)}
    .rerank-result-head{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:12px}
    .rerank-result-text{margin-top:8px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55}
    .card,.metric,.kv-item,.sidebar-card{min-width:0}
    .card-body{padding:18px}
    .card-title{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:0 0 14px}
    .card-title h3{margin:0;font-size:18px}
    .card-title p{margin:6px 0 0;color:var(--muted);font-size:13px}
    .status{
      display:flex;align-items:center;gap:10px;padding:12px;border-radius:10px;
      background:rgba(15,23,42,.46);border:1px solid var(--border);font-size:13px
    }
    .light{width:10px;height:10px;border-radius:999px;background:var(--muted)}
    .status.ok .light{background:var(--good);box-shadow:0 0 0 5px rgba(34,197,94,.15)}
    .status.warn .light{background:var(--warn);box-shadow:0 0 0 5px rgba(245,158,11,.16)}
    .status.bad .light{background:var(--danger);box-shadow:0 0 0 5px rgba(239,68,68,.15)}
    .error-box{
      display:none;margin-top:14px;padding:12px;border-radius:10px;
      border:1px solid rgba(239,68,68,.28);background:rgba(127,29,29,.22);color:#fecaca
    }
    .error-box.show{display:block}
    .muted{color:var(--muted)}
    .hint{margin-top:8px;font-size:12px;color:var(--muted)}
    label{display:block;margin:0 0 8px;font-size:13px;color:#dbeafe}
    textarea,input,select{
      width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--border);
      background:rgba(15,23,42,.62);color:var(--text);font:inherit
    }
    textarea{min-height:220px;resize:vertical}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
    .actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
    .btn{
      appearance:none;border:none;border-radius:10px;padding:10px 13px;
      font:600 13px/1 inherit;cursor:pointer;transition:transform .06s,opacity .16s
    }
    .btn:hover{opacity:.94}
    .btn:active{transform:translateY(1px)}
    .btn:disabled{opacity:.55;cursor:not-allowed}
    .btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#111827}
    .btn.secondary{background:rgba(15,23,42,.64);border:1px solid var(--border);color:var(--text)}
    .btn.ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
    .metric{
      padding:12px;border-radius:10px;border:1px solid var(--border);background:rgba(15,23,42,.48)
    }
    .metric .n{font-size:24px;font-weight:700}
    .metric .l{font-size:12px;color:var(--muted)}
    pre{
      margin:0;white-space:pre-wrap;word-break:break-word;background:rgba(2,6,23,.72);
      border:1px solid var(--border);padding:12px;border-radius:10px;max-height:420px;overflow:auto
    }
    .toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}
    .toolbar .small{font-size:12px;color:var(--muted)}
    .matrix-wrap{overflow:auto;border:1px solid var(--border);border-radius:10px;background:rgba(2,6,23,.52)}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;border-bottom:1px solid rgba(148,163,184,.08);text-align:center;font-size:12px}
    th:first-child,td:first-child{text-align:left;color:var(--muted)}
    .pill{
      display:inline-flex;padding:6px 10px;border-radius:999px;border:1px solid var(--border);
      background:rgba(255,255,255,.06);font-size:12px;color:#dbeafe
    }
    .templates{display:flex;flex-direction:column;gap:10px}
    .template-note{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:rgba(15,23,42,.42);font-size:13px;color:var(--muted)}
    .kv-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
    .kv-item{border:1px solid var(--border);background:rgba(15,23,42,.44);border-radius:10px;padding:10px}
    .kv-item span{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
    .kv-item strong{font-size:15px;font-weight:600;color:var(--text)}
    .api-list{list-style:none;padding:0;margin:0;display:grid;gap:8px}
    .api-list li{border:1px solid var(--border);border-radius:10px;background:rgba(15,23,42,.42);padding:10px 12px}
    .api-list p{margin:6px 0 0;color:var(--muted);font-size:12px}
    .inline-code{font-family:ui-monospace, SFMono-Regular, Menlo, monospace}
    @media (max-width: 920px){
      .app{display:block}
      .sidebar{
        width:auto;
        height:auto;
        position:relative;
        top:auto;
        border:none;
        border-bottom:1px solid var(--border);
        border-radius:0;
        overflow:visible;
      }
      .grid,.grid-2,.row,.row-3,.metrics{grid-template-columns:1fr}
      .kv-grid{grid-template-columns:1fr}
    }
  </style>
  <style id="unified-ui-overrides">
    :root{
      --border2:rgba(148,163,184,.24);
      --ring:0 0 0 4px rgba(255,138,31,.16);
      --radius:8px;
      --radius2:8px;
    }
    *{
      scrollbar-width:thin;
      scrollbar-color:rgba(148,163,184,.34) transparent;
    }
    *::-webkit-scrollbar{width:10px;height:10px}
    *::-webkit-scrollbar-track{background:transparent}
    *::-webkit-scrollbar-thumb{
      border:2px solid transparent;
      border-radius:999px;
      background:rgba(148,163,184,.34);
      background-clip:padding-box;
    }
    *::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,.48);background-clip:padding-box}
    html{min-height:100%;background:var(--bg0)}
    body{
      min-height:100%;
      position:relative;
      background:linear-gradient(180deg,#0a1224 0%,#08101f 48%,#050913 100%);
    }
    body::before{
      content:"";
      position:fixed;
      inset:0;
      z-index:0;
      pointer-events:none;
      background:
        radial-gradient(960px 560px at 18% -12%,rgba(255,138,31,.14),transparent 64%),
        radial-gradient(820px 520px at 92% 8%,rgba(255,209,102,.08),transparent 62%);
    }
    .app{position:relative;z-index:1;display:flex;min-height:100vh;padding:0;gap:0;align-items:stretch}
    .sidebar{
      width:270px;
      min-width:270px;
      height:100vh;
      max-height:100vh;
      position:sticky;
      top:0;
      padding:18px 16px;
      overflow:auto;
      border:0;
      border-right:1px solid var(--border);
      border-radius:0;
      background:rgba(2,6,23,.58);
      backdrop-filter:blur(10px);
    }
    .brand{gap:12px;padding:10px 10px 14px}
    .logo{
      width:34px;
      height:34px;
      border-radius:0;
      object-fit:contain;
      box-shadow:none;
    }
    .brand-title{font-size:16px;font-weight:700}
    .brand-sub{font-size:12px;color:var(--muted2);margin-top:2px}
    .nav{margin-top:6px;display:flex;flex-direction:column;gap:6px}
    .nav .nav-link{
      display:flex;
      align-items:center;
      gap:10px;
      width:100%;
      padding:10px 12px;
      border:1px solid transparent;
      border-radius:6px;
      background:transparent;
      color:var(--muted);
      font:inherit;
      text-align:left;
      cursor:pointer;
      transition:background .15s ease,border-color .15s ease,color .15s ease;
    }
    .nav .nav-link:hover{background:rgba(148,163,184,.08)}
    .nav .nav-link.active{
      border-color:rgba(255,138,31,.24);
      background:rgba(255,138,31,.11);
      color:var(--text);
    }
    .icon-defs{position:absolute;width:0;height:0;overflow:hidden}
    .icon{
      width:16px;
      height:16px;
      flex:0 0 16px;
      color:currentColor;
      fill:none;
      stroke:currentColor;
      stroke-width:1.8;
      stroke-linecap:square;
      stroke-linejoin:miter;
      opacity:.84;
    }
    .nav .icon{color:var(--muted2);opacity:.78}
    .nav .nav-link.active .icon{color:var(--accent);opacity:.98}
    .sidebar-footer{
      margin-top:18px;
      padding:12px;
      border:1px solid var(--border);
      border-radius:var(--radius);
      background:rgba(15,23,42,.45);
    }
    .sidebar-footer .kv{display:flex;justify-content:space-between;gap:10px;padding:6px 0}
    .sidebar-footer .k{color:var(--muted2);font-size:12px}
    .sidebar-footer .v{
      max-width:142px;
      overflow:hidden;
      color:var(--text);
      font-size:12px;
      text-align:right;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .main{min-width:0;flex:1;display:flex;flex-direction:column}
    .topbar{
      position:sticky;
      top:0;
      z-index:12;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding:12px 18px;
      border-bottom:1px solid var(--border);
      background:rgba(2,6,23,.58);
      backdrop-filter:blur(10px);
    }
    .crumbs{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted)}
    .crumbs strong{color:var(--text)}
    .sep{opacity:.55}
    .top-actions{display:flex;align-items:center;justify-content:flex-end;gap:10px;flex-wrap:wrap}
    .chips{display:flex;gap:10px;flex-wrap:wrap}
    .chip{padding:8px 12px;font-weight:650;white-space:nowrap}
    .content{padding:18px 16px 34px}
    .tabbar{display:none}
    .mobile-nav{display:none}
    .section{max-width:1320px;margin:0 auto 16px;display:none}
    .section.active{display:block}
    .section.projector-section{max-width:1600px}
    .section-head{margin:2px 0 12px}
    .section-head h1,.section-head h2{margin:0;font-size:22px;letter-spacing:.2px}
    .section-head p{margin:8px 0 0;color:var(--muted);font-size:13px;line-height:1.7}
    .grid{grid-template-columns:1.12fr .88fr;gap:10px}
    .grid-2{gap:10px}
    .card{
      border:1px solid var(--border);
      border-radius:var(--radius2);
      background:var(--panel);
      box-shadow:0 12px 34px rgba(0,0,0,.30);
      overflow:hidden;
    }
    .card-body{padding:12px 14px}
    .card-title{
      min-height:45px;
      margin:-12px -14px 12px;
      padding:10px 14px;
      align-items:center;
      border-bottom:1px solid rgba(148,163,184,.20);
      background:linear-gradient(180deg,rgba(30,41,59,.54),rgba(15,23,42,.26));
    }
    .card-title h3{font-size:13px;font-weight:760;letter-spacing:.24px;color:rgba(241,245,249,.96)}
    .card-title p{margin:4px 0 0;font-size:12px;line-height:1.5}
    .pill{padding:5px 9px;border-radius:999px}
    label{margin:0 0 7px;font-size:12px;color:var(--muted2)}
    textarea,input,select{
      padding:10px 11px;
      border-radius:6px;
      border-color:var(--border);
      background:rgba(2,6,23,.38);
      outline:none;
      transition:border-color .15s ease,box-shadow .15s ease,background .15s ease;
    }
    textarea::placeholder,input::placeholder{color:rgba(148,163,184,.55)}
    textarea:focus,input:focus,select:focus{
      border-color:rgba(255,138,31,.44);
      box-shadow:var(--ring);
      background:rgba(2,6,23,.46);
    }
    textarea{min-height:230px}
    .row,.row-3{gap:10px}
    .actions{gap:10px;margin-top:14px}
    .btn{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:8px;
      padding:10px 12px;
      border:1px solid var(--border);
      border-radius:6px;
      font:600 13px/1 inherit;
      transition:background .15s ease,border-color .15s ease,transform .05s ease,opacity .15s ease;
    }
    .btn:hover{opacity:1;background:rgba(148,163,184,.10)}
    .btn.primary{
      border-color:rgba(255,138,31,.28);
      background:linear-gradient(180deg,rgba(255,138,31,.28),rgba(255,138,31,.17));
      color:rgba(255,244,226,.96);
    }
    .btn.primary:hover{background:linear-gradient(180deg,rgba(255,138,31,.34),rgba(255,138,31,.20))}
    .btn.secondary{background:rgba(15,23,42,.48)}
    .btn.ghost{background:transparent}
    .btn:focus-visible,.nav-link:focus-visible,.mobile-tab:focus-visible{
      outline:none;
      border-color:rgba(255,138,31,.48);
      box-shadow:var(--ring);
    }
    .status{padding:11px;border-radius:6px;background:rgba(2,6,23,.30)}
    .metrics{gap:10px}
    .metric,.kv-item,.template-note,.api-list li{border-radius:6px;background:rgba(2,6,23,.24)}
    .metric .n{font-size:22px}
    pre,.matrix-wrap{border-radius:6px;background:rgba(2,6,23,.46)}
    .embedding-viz{
      margin-top:12px;
      padding:12px;
      border:1px solid var(--border);
      border-radius:8px;
      background:rgba(2,6,23,.24);
    }
    .embedding-viz-head{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:12px;
      margin-bottom:10px;
    }
    .embedding-viz-title{font-size:13px;font-weight:700;color:var(--text)}
    .embedding-viz-subtitle{margin-top:3px;font-size:12px;color:var(--muted2)}
    .similarity-empty{
      display:flex;
      align-items:center;
      justify-content:center;
      min-height:176px;
      padding:16px;
      border:1px dashed rgba(148,163,184,.22);
      border-radius:6px;
      color:var(--muted2);
      font-size:12px;
      text-align:center;
    }
    .similarity-grid-wrap{overflow:auto}
    .similarity-grid{
      display:grid;
      gap:5px;
      min-width:max-content;
      align-items:stretch;
    }
    .similarity-axis,
    .similarity-cell{
      display:flex;
      align-items:center;
      justify-content:center;
      min-height:38px;
      padding:6px 7px;
      border:1px solid rgba(148,163,184,.13);
      border-radius:5px;
      font-size:11px;
      font-variant-numeric:tabular-nums;
    }
    .similarity-axis{background:rgba(15,23,42,.46);color:var(--muted);font-weight:650}
    .similarity-axis.row-label{
      justify-content:flex-start;
      max-width:150px;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .similarity-cell{color:rgba(241,245,249,.94);background:rgba(255,138,31,var(--cell-alpha,.12))}
    .similarity-cell.negative{background:rgba(96,165,250,var(--cell-alpha,.12))}
    .similarity-legend{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      margin-top:8px;
      color:var(--muted2);
      font-size:11px;
    }
    .similarity-gradient{
      width:120px;
      height:7px;
      border-radius:999px;
      background:linear-gradient(90deg,rgba(96,165,250,.65),rgba(148,163,184,.12),rgba(255,138,31,.72));
    }
    .vector-profile{margin-top:12px;padding-top:10px;border-top:1px solid rgba(148,163,184,.12)}
    .vector-profile-head{
      display:flex;
      justify-content:space-between;
      gap:10px;
      margin-bottom:7px;
      color:var(--muted2);
      font-size:11px;
    }
    .vector-profile svg{
      display:block;
      width:100%;
      height:76px;
      overflow:visible;
      border:1px solid rgba(148,163,184,.12);
      border-radius:5px;
      background:rgba(2,6,23,.34);
    }
    .payload-details{margin-top:10px;border:1px solid var(--border);border-radius:6px;background:rgba(2,6,23,.20)}
    .payload-details summary{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      padding:10px 11px;
      color:var(--muted);
      font-size:12px;
      cursor:pointer;
      user-select:none;
    }
    .payload-details[open] summary{border-bottom:1px solid rgba(148,163,184,.12)}
    .payload-details pre{margin:10px;max-height:320px}
    .projector-loading{
      display:flex;
      align-items:center;
      gap:10px;
      min-height:220px;
      padding:20px;
      border:1px solid var(--border);
      border-radius:8px;
      background:var(--panel);
      color:var(--muted);
    }
    .loading-dot{
      width:10px;
      height:10px;
      flex:0 0 10px;
      border-radius:999px;
      background:var(--accent);
      box-shadow:0 0 0 5px rgba(255,138,31,.14);
      animation:pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse{0%,100%{opacity:.42;transform:scale(.82)}50%{opacity:1;transform:scale(1)}}
    @media (max-width:1180px){
      .topbar{align-items:flex-start}
      .metrics{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:980px){
      .app{display:block}
      .sidebar{display:none}
      .topbar{padding:10px 12px}
      .topbar .chip.optional{display:none}
      .content{padding:10px 10px 28px}
      .mobile-nav{
        position:sticky;
        top:59px;
        z-index:9;
        display:flex;
        gap:6px;
        margin:0 0 12px;
        padding:6px;
        overflow-x:auto;
        border:1px solid var(--border);
        border-radius:8px;
        background:rgba(2,6,23,.78);
        backdrop-filter:blur(10px);
      }
      .mobile-tab{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        gap:7px;
        flex:1 0 auto;
        padding:9px 11px;
        border:1px solid transparent;
        border-radius:6px;
        background:transparent;
        color:var(--muted);
        font:600 12px/1 inherit;
        cursor:pointer;
      }
      .mobile-tab.active{border-color:rgba(255,138,31,.24);background:rgba(255,138,31,.11);color:var(--text)}
      .grid,.grid-2,.row,.row-3,.kv-grid{grid-template-columns:1fr}
      .section,.section.projector-section{max-width:none}
    }
    @media (max-width:640px){
      .topbar{align-items:center}
      .topbar .chips{display:none}
      .top-actions .btn span{display:none}
      .section-head h1,.section-head h2{font-size:20px}
      .metrics{grid-template-columns:1fr 1fr}
      textarea{min-height:190px}
    }
    @media (prefers-reduced-motion:reduce){
      *{transition:none!important}
      .loading-dot{animation:none}
    }
  </style>
</head>
<body>
  <svg class="icon-defs" aria-hidden="true" focusable="false">
    <symbol id="i-embed" viewBox="0 0 24 24">
      <circle cx="6" cy="12" r="2"/><circle cx="17" cy="6" r="2"/><circle cx="18" cy="17" r="2"/>
      <path d="M8 11l7-4M8 13l8 3M17 8l1 7"/>
    </symbol>
    <symbol id="i-results" viewBox="0 0 24 24">
      <path d="M5 19V9M12 19V5M19 19v-7M3 19h18"/>
    </symbol>
    <symbol id="i-projector" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="2"/><circle cx="5" cy="7" r="1.5"/><circle cx="18" cy="5" r="1.5"/><circle cx="19" cy="18" r="1.5"/>
      <path d="M7 8l3.5 3M13.5 10.5L17 6.5M13.5 13.5l4 3.5"/>
    </symbol>
    <symbol id="i-admin" viewBox="0 0 24 24">
      <path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M7 14v6"/>
    </symbol>
    <symbol id="i-doc" viewBox="0 0 24 24">
      <path d="M7 4h7l4 4v12H7zM14 4v4h4M10 12h5M10 16h5"/>
    </symbol>
    <symbol id="i-health" viewBox="0 0 24 24">
      <path d="M4 13h4l2-5 4 10 2-5h4"/>
    </symbol>
    <symbol id="i-run" viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></symbol>
    <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M19 8V4l-2 2a7 7 0 10 1.5 9M19 4h-4"/></symbol>
    <symbol id="i-copy" viewBox="0 0 24 24"><path d="M9 9h10v11H9zM6 16H5V5h11v1"/></symbol>
    <symbol id="i-download" viewBox="0 0 24 24"><path d="M12 4v11M8 11l4 4 4-4M5 20h14"/></symbol>
    <symbol id="i-demo" viewBox="0 0 24 24"><path d="M5 19h14M7 16l3-8 3 5 2-7 2 10"/></symbol>
    <symbol id="i-reload" viewBox="0 0 24 24"><path d="M18 8V4l-2 2a7 7 0 11-1-1M18 4h-4"/></symbol>
    <symbol id="i-template" viewBox="0 0 24 24"><path d="M5 5h6v6H5zM13 5h6v6h-6zM5 13h6v6H5zM13 13h6v6h-6z"/></symbol>
  </svg>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <img class="logo" src="/static/logo.png" alt="Qwen3 Embedding and Reranker"/>
        <div>
          <div class="brand-title">Qwen3 Embedding &amp; Reranker</div>
          <div class="brand-sub">Self-hosted Debug Console</div>
        </div>
      </div>

      <nav class="nav" aria-label="主导航">
        <button type="button" class="nav-link active" data-tab="debug-section"><svg class="icon" aria-hidden="true"><use href="#i-embed"></use></svg><span>调试台</span></button>
        <button type="button" class="nav-link" data-tab="reranker-section"><svg class="icon" aria-hidden="true"><use href="#i-results"></use></svg><span>Reranker</span></button>
        <button type="button" class="nav-link" data-tab="results-section"><svg class="icon" aria-hidden="true"><use href="#i-results"></use></svg><span>结果分析</span></button>
        <button type="button" class="nav-link" data-tab="projector-section"><svg class="icon" aria-hidden="true"><use href="#i-projector"></use></svg><span>Projector 视图</span></button>
        <button type="button" class="nav-link" data-tab="admin-section"><svg class="icon" aria-hidden="true"><use href="#i-admin"></use></svg><span>运维管理</span></button>
        <a class="nav-link" href="/docs" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg><span>API 文档</span></a>
        <a class="nav-link" href="/health" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-health"></use></svg><span>健康检查</span></a>
      </nav>

      <div class="sidebar-footer" aria-label="运行信息">
        <div class="kv"><span class="k">Endpoint</span><span class="v">/v1/embeddings</span></div>
        <div class="kv"><span class="k">Rerank</span><span class="v">/v1/rerank</span></div>
        <div class="kv"><span class="k">MCP</span><span class="v">/mcp</span></div>
        <div class="kv"><span class="k">Model</span><span class="v" id="sidebarModel">__MODEL_ID__</span></div>
        <div class="kv"><span class="k">Device</span><span class="v" id="sidebarDevice">cuda</span></div>
        <div class="kv"><span class="k">DType</span><span class="v" id="sidebarDtype">—</span></div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div class="crumbs"><span class="muted2">Services</span><span class="sep">/</span><strong id="activeViewLabel">Embedding 调试台</strong></div>
        <div class="top-actions">
          <div class="chips">
            <span class="chip optional" id="topModelChip">Model: __MODEL_ID__</span>
            <span class="chip warn" id="topStatusChip">Backend: checking</span>
            <span class="chip optional" id="topDeviceChip">Target: CUDA</span>
          </div>
          <a class="btn ghost" href="/docs" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg><span>Docs</span></a>
          <a class="btn ghost" href="/redoc" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg><span>Redoc</span></a>
        </div>
      </header>

      <div class="content">
        <nav class="mobile-nav" aria-label="移动端导航">
          <button type="button" class="mobile-tab active" data-tab="debug-section"><svg class="icon" aria-hidden="true"><use href="#i-embed"></use></svg>调试台</button>
          <button type="button" class="mobile-tab" data-tab="reranker-section"><svg class="icon" aria-hidden="true"><use href="#i-results"></use></svg>Rerank</button>
          <button type="button" class="mobile-tab" data-tab="results-section"><svg class="icon" aria-hidden="true"><use href="#i-results"></use></svg>结果</button>
          <button type="button" class="mobile-tab" data-tab="projector-section"><svg class="icon" aria-hidden="true"><use href="#i-projector"></use></svg>Projector</button>
          <button type="button" class="mobile-tab" data-tab="admin-section"><svg class="icon" aria-hidden="true"><use href="#i-admin"></use></svg>运维</button>
        </nav>

        <section class="section active tab-panel" id="debug-section">
          <div class="section-head">
            <h1>Embedding 调试台</h1>
            <p>生成与检查文本向量，查看运行状态，并通过模板、向量下载和相似度矩阵完成接入调试。</p>
          </div>

          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Generate Embeddings</h3>
                    <p>输入文本后直接调用 <code>/v1/embeddings</code>。两条以上文本会自动计算相似度矩阵。</p>
                  </div>
                  <span class="pill" id="templateChip">Template: custom</span>
                </div>

                <div class="row">
                  <div>
                    <label for="templateSelect">请求模板</label>
                    <select id="templateSelect">
                      <option value="blank">空白模板</option>
                      <option value="retrieval_pair">检索对比</option>
                      <option value="query_only">单条查询</option>
                      <option value="document_batch">文档批量</option>
                      <option value="news_similarity">新闻相似度</option>
                    </select>
                  </div>
                  <div>
                    <label>&nbsp;</label>
                    <button class="btn secondary" id="applyTemplateBtn" style="width:100%"><svg class="icon" aria-hidden="true"><use href="#i-template"></use></svg>应用模板</button>
                  </div>
                </div>

                <label for="texts" style="margin-top:16px">Texts</label>
                <textarea id="texts" placeholder="每行一条文本。"></textarea>
                <div class="hint">输入会按非空行拆分。若选择 <code>query</code>，服务会在每条文本前自动拼接 instruction。</div>

                <div class="row" style="margin-top:16px">
                  <div>
                    <label for="inputType">Input Type</label>
                    <select id="inputType">
                      <option value="">document / raw</option>
                      <option value="query">query</option>
                      <option value="document">document</option>
                    </select>
                  </div>
                  <div>
                    <label for="dimensions">Dimensions</label>
                    <input id="dimensions" type="number" min="32" placeholder="例如 1024（上限随模型）"/>
                  </div>
                </div>

                <div class="row" style="margin-top:12px">
                  <div>
                    <label for="instruction">Instruction</label>
                    <input id="instruction" placeholder="留空则使用默认 query instruction"/>
                  </div>
                  <div>
                    <label for="encodingFormat">Encoding Format</label>
                    <select id="encodingFormat">
                      <option value="">float</option>
                      <option value="float">float</option>
                      <option value="base64">base64</option>
                    </select>
                  </div>
                </div>

                <div class="actions">
                  <button class="btn primary" id="runBtn"><svg class="icon" aria-hidden="true"><use href="#i-run"></use></svg>生成 Embedding</button>
                  <button class="btn secondary" id="healthBtn"><svg class="icon" aria-hidden="true"><use href="#i-refresh"></use></svg>刷新状态</button>
                  <button class="btn secondary" id="copyJsonBtn"><svg class="icon" aria-hidden="true"><use href="#i-copy"></use></svg>复制 JSON</button>
                  <button class="btn secondary" id="downloadBtn"><svg class="icon" aria-hidden="true"><use href="#i-download"></use></svg>下载向量</button>
                  <button class="btn ghost" id="fillDemoBtn"><svg class="icon" aria-hidden="true"><use href="#i-demo"></use></svg>填充 Demo</button>
                </div>
              </div>
            </section>

            <section class="card" id="runtime-section">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Runtime</h3>
                    <p>后端健康、模型、时区、显卡目标和当前错误都会在这里聚合展示。</p>
                  </div>
                </div>

                <div class="status warn" id="statusBox"><span class="light"></span><span id="statusText">Checking backend...</span></div>
                <div class="error-box" id="errorBox">
                  <strong>最近错误</strong>
                  <div id="errorText" style="margin-top:8px"></div>
                </div>

                <div class="metrics">
                  <div class="metric"><div class="n" id="metricCount">0</div><div class="l">Vectors</div></div>
                  <div class="metric"><div class="n" id="metricDim">-</div><div class="l">Dimensions</div></div>
                  <div class="metric"><div class="n" id="metricLatency">-</div><div class="l">Latency</div></div>
                  <div class="metric"><div class="n" id="metricState">-</div><div class="l">Backend State</div></div>
                </div>

                <div class="kv-grid">
                  <div class="kv-item"><span>模型</span><strong id="healthModelChip">__MODEL_ID__</strong></div>
                  <div class="kv-item"><span>设备目标</span><strong id="healthDeviceChip">cuda</strong></div>
                </div>

                <div class="embedding-viz" aria-live="polite">
                  <div class="embedding-viz-head">
                    <div>
                      <div class="embedding-viz-title">Embedding 可视化</div>
                      <div class="embedding-viz-subtitle">余弦相似度与首个向量的维度采样轮廓</div>
                    </div>
                    <span class="pill">cosine</span>
                  </div>
                  <div id="similarityHeatmap" class="similarity-empty">生成 Embedding 后在这里显示语义关系。</div>
                </div>

                <details class="payload-details">
                  <summary><span>Health Payload</span><span class="muted2">JSON</span></summary>
                  <pre id="healthOut">loading...</pre>
                </details>
              </div>
            </section>
          </div>
        </section>

        <section class="section tab-panel" id="reranker-section">
          <div class="section-head">
            <h2>Reranker 调试台</h2>
            <p>对第一阶段召回的候选文档进行精排。每行一篇文档，最多 50 篇；instruction 与模板固定在服务端。</p>
          </div>
          <div class="grid">
            <section class="card">
              <div class="card-body">
                <label for="rerankQuery">Query</label>
                <input id="rerankQuery" value="中国的首都是哪里？"/>
                <label for="rerankDocuments" style="margin-top:14px">Documents</label>
                <textarea id="rerankDocuments" placeholder="每行一篇候选文档。">巴黎是法国的首都。\n北京是中华人民共和国的首都。\nPython 使用缩进表示代码块。</textarea>
                <div class="row" style="margin-top:12px">
                  <div>
                    <label for="rerankTopN">Top N</label>
                    <input id="rerankTopN" type="number" min="1" max="50" placeholder="留空返回全部"/>
                  </div>
                  <div>
                    <label>固定模型</label>
                    <input value="__RERANKER_MODEL_ID__" disabled/>
                  </div>
                </div>
                <div class="actions">
                  <button class="btn primary" id="rerankBtn"><svg class="icon" aria-hidden="true"><use href="#i-run"></use></svg>执行 Rerank</button>
                </div>
              </div>
            </section>
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div><h3>排序结果</h3><p id="rerankMeta">尚未请求。</p></div>
                </div>
                <div class="rerank-results" id="rerankResults"></div>
                <details class="payload-details" open>
                  <summary><span>Rerank Response</span><span class="muted2">JSON</span></summary>
                  <pre id="rerankRawOut">尚未请求。</pre>
                </details>
              </div>
            </section>
          </div>
        </section>

        <section class="section tab-panel" id="results-section">
          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>结果概览</h3>
                    <p>向量数量、维度、token 统计和前几维预览。</p>
                  </div>
                </div>
                <pre id="summaryOut">尚未请求。</pre>
              </div>
            </section>

            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>模板说明</h3>
                    <p>常见调试场景可以一键填充，避免重复手输。</p>
                  </div>
                </div>
                <div class="templates" id="templateHelp">
                  <div class="template-note">选择模板后会自动填充 <code>texts</code>、<code>input_type</code>、<code>instruction</code> 和 <code>dimensions</code>。</div>
                </div>
              </div>
            </section>
          </div>

          <div class="grid" style="margin-top:16px">
            <section class="card">
              <div class="card-body">
                <div class="toolbar">
                  <div>
                    <h3 style="margin:0">Similarity Matrix</h3>
                    <div class="small">最多展示当前请求的前 8 条文本。</div>
                  </div>
                </div>
                <div class="matrix-wrap" id="matrixWrap">
                  <table id="matrixTable">
                    <tbody><tr><td style="padding:16px;color:var(--muted)">尚未请求。</td></tr></tbody>
                  </table>
                </div>
              </div>
            </section>

            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Raw Response</h3>
                    <p>直接展示 `/v1/embeddings` 的原始 JSON 响应，便于复制和比对。</p>
                  </div>
                </div>
                <pre id="rawOut">尚未请求。</pre>
              </div>
            </section>
          </div>
        </section>

        <section class="section projector-section tab-panel" id="projector-section">
          <div class="section-head">
            <h2>Embedding Projector</h2>
            <p>在三维空间中观察文本向量分布，并联动检查选中点、最近邻和投影元数据。</p>
          </div>
          <div id="projector-root">
            <div class="projector-loading" id="projectorLoading">
              <span class="loading-dot" aria-hidden="true"></span>
              <span>首次打开时加载 Projector 可视化模块。</span>
            </div>
          </div>
        </section>

        <section class="section tab-panel" id="admin-section">
          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Embedding 热重载</h3>
                    <p>仅重启 Embedding 后端；Reranker 不受影响。</p>
                  </div>
                </div>

                <div class="row">
                  <div>
                    <label for="reloadModelId">Model ID</label>
                    <input id="reloadModelId" placeholder="Qwen/Qwen3-Embedding-4B"/>
                  </div>
                  <div>
                    <label for="reloadToken">x-admin-token</label>
                    <input id="reloadToken" placeholder="change-me"/>
                  </div>
                </div>

                <div class="row-3" style="margin-top:12px">
                  <div>
                    <label for="reloadDtype">DType</label>
                    <input id="reloadDtype" placeholder="float16"/>
                  </div>
                  <div>
                    <label for="reloadMaxModelLen">Max Model Len</label>
                    <input id="reloadMaxModelLen" type="number" placeholder="4096"/>
                  </div>
                  <div>
                    <label for="reloadGpuUtil">GPU Memory Utilization</label>
                    <input id="reloadGpuUtil" type="number" step="0.01" min="0.1" max="1" placeholder="0.72"/>
                  </div>
                </div>

                <div class="row" style="margin-top:12px">
                  <div>
                    <label for="reloadInstruction">Default Query Instruction</label>
                    <input id="reloadInstruction" placeholder="Given a web search query, retrieve relevant passages that answer the query"/>
                  </div>
                  <div>
                    <label for="reloadExtraArgs">VLLM Extra Args</label>
                    <input id="reloadExtraArgs" placeholder="--tensor-parallel-size 2"/>
                  </div>
                </div>

                <div class="actions">
                  <button class="btn primary" id="reloadBtn"><svg class="icon" aria-hidden="true"><use href="#i-reload"></use></svg>提交热重载</button>
                </div>

                <div style="margin-top:16px">
                  <label>Reload Response</label>
                  <pre id="reloadOut">尚未提交。</pre>
                </div>
              </div>
            </section>

            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Reranker 状态与热重载</h3>
                    <p>只允许修改 revision 或在 FP16 / BitsAndBytes 4-bit 之间切换，显存与并发上限不可在线放宽。</p>
                  </div>
                </div>
                <div class="row">
                  <div>
                    <label for="rerankerRevision">Model Revision</label>
                    <input id="rerankerRevision" placeholder="留空保持当前 revision"/>
                  </div>
                  <div>
                    <label for="rerankerQuantization">Quantization</label>
                    <select id="rerankerQuantization">
                      <option value="none">none / FP16</option>
                      <option value="bitsandbytes">bitsandbytes 4-bit</option>
                    </select>
                  </div>
                </div>
                <div class="actions">
                  <button class="btn primary" id="rerankerReloadBtn"><svg class="icon" aria-hidden="true"><use href="#i-reload"></use></svg>重载 Reranker</button>
                </div>
                <pre id="rerankerReloadOut" style="margin-top:16px">尚未提交。</pre>
              </div>
            </section>

            <section class="card" id="api-section">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>接口说明</h3>
                    <p>当前页面只是内置调试台，正式接入仍建议直接调用 API。</p>
                  </div>
                </div>
                <ul class="api-list">
                  <li><code>POST /v1/embeddings</code><p>OpenAI 兼容 Embeddings 接口</p></li>
                  <li><code>POST /v1/rerank</code><p>vLLM 兼容的二阶段文档精排接口</p></li>
                  <li><code>POST /v1/embeddings/projector</code><p>后端预计算 3D 投影 + 近邻，供 Projector 视图渲染</p></li>
                  <li><code>GET /</code><p>内置调试页面</p></li>
                  <li><code>GET /projector</code><p>兼容入口，跳转到主页 Projector 页签</p></li>
                  <li><code>POST/GET /mcp</code><p>MCP Streamable HTTP 入口</p></li>
                  <li><code>GET /health</code><p>运行状态与 backend 聚合健康</p></li>
                  <li><code>POST /admin/reranker/reload</code><p>独立重载 Reranker</p></li>
                  <li><code>GET /docs</code> / <code>GET /redoc</code><p>Swagger 与 ReDoc 文档</p></li>
                </ul>
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const REQUEST_TIMEOUT_MS = 20000;
    let lastHealthPayload = null;
    let lastEmbeddingPayload = null;
    let lastRerankPayload = null;

    const templates = {
      blank: {
        label: "空白模板",
        note: "保留当前输入，只重置为最基础的 document/raw 调试模式。",
        texts: "",
        inputType: "",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
      retrieval_pair: {
        label: "检索对比",
        note: "最适合验证 query/document 语义是否正确，以及相似度是否符合预期。",
        texts: "What is the capital of China?\\nThe capital of China is Beijing.\\nThe Eiffel Tower is located in Paris.",
        inputType: "query",
        instruction: "__DEFAULT_INSTRUCTION__",
        dimensions: "",
        encodingFormat: "float",
      },
      query_only: {
        label: "单条查询",
        note: "用于验证 query 侧 instruction 注入是否成功。",
        texts: "How to improve retrieval quality for Chinese documents?",
        inputType: "query",
        instruction: "__DEFAULT_INSTRUCTION__",
        dimensions: "",
        encodingFormat: "float",
      },
      document_batch: {
        label: "文档批量",
        note: "把多条文档作为 document/raw 一次性编码，并查看相似度矩阵。",
        texts: "Beijing is the capital of China.\\nShanghai is a major financial center in China.\\nParis is the capital of France.",
        inputType: "document",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
      news_similarity: {
        label: "新闻相似度",
        note: "用于比较多段新闻文本之间的语义距离。",
        texts: "中国央行宣布新的利率政策。\\n央行今日发布利率调整公告。\\n足球联赛将在本周末开赛。",
        inputType: "document",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
    };

    const els = {
      texts: document.getElementById("texts"),
      inputType: document.getElementById("inputType"),
      dimensions: document.getElementById("dimensions"),
      instruction: document.getElementById("instruction"),
      encodingFormat: document.getElementById("encodingFormat"),
      templateSelect: document.getElementById("templateSelect"),
      applyTemplateBtn: document.getElementById("applyTemplateBtn"),
      templateChip: document.getElementById("templateChip"),
      templateHelp: document.getElementById("templateHelp"),
      runBtn: document.getElementById("runBtn"),
      fillDemoBtn: document.getElementById("fillDemoBtn"),
      healthBtn: document.getElementById("healthBtn"),
      copyJsonBtn: document.getElementById("copyJsonBtn"),
      downloadBtn: document.getElementById("downloadBtn"),
      statusBox: document.getElementById("statusBox"),
      statusText: document.getElementById("statusText"),
      errorBox: document.getElementById("errorBox"),
      errorText: document.getElementById("errorText"),
      metricCount: document.getElementById("metricCount"),
      metricDim: document.getElementById("metricDim"),
      metricLatency: document.getElementById("metricLatency"),
      metricState: document.getElementById("metricState"),
      similarityHeatmap: document.getElementById("similarityHeatmap"),
      healthOut: document.getElementById("healthOut"),
      summaryOut: document.getElementById("summaryOut"),
      rawOut: document.getElementById("rawOut"),
      matrixTable: document.getElementById("matrixTable"),
      topModelChip: document.getElementById("topModelChip"),
      topStatusChip: document.getElementById("topStatusChip"),
      topDeviceChip: document.getElementById("topDeviceChip"),
      activeViewLabel: document.getElementById("activeViewLabel"),
      sidebarModel: document.getElementById("sidebarModel"),
      sidebarDevice: document.getElementById("sidebarDevice"),
      sidebarDtype: document.getElementById("sidebarDtype"),
      healthModelChip: document.getElementById("healthModelChip"),
      healthDeviceChip: document.getElementById("healthDeviceChip"),
      reloadModelId: document.getElementById("reloadModelId"),
      reloadToken: document.getElementById("reloadToken"),
      reloadDtype: document.getElementById("reloadDtype"),
      reloadMaxModelLen: document.getElementById("reloadMaxModelLen"),
      reloadGpuUtil: document.getElementById("reloadGpuUtil"),
      reloadInstruction: document.getElementById("reloadInstruction"),
      reloadExtraArgs: document.getElementById("reloadExtraArgs"),
      reloadBtn: document.getElementById("reloadBtn"),
      reloadOut: document.getElementById("reloadOut"),
      rerankQuery: document.getElementById("rerankQuery"),
      rerankDocuments: document.getElementById("rerankDocuments"),
      rerankTopN: document.getElementById("rerankTopN"),
      rerankBtn: document.getElementById("rerankBtn"),
      rerankResults: document.getElementById("rerankResults"),
      rerankMeta: document.getElementById("rerankMeta"),
      rerankRawOut: document.getElementById("rerankRawOut"),
      rerankerRevision: document.getElementById("rerankerRevision"),
      rerankerQuantization: document.getElementById("rerankerQuantization"),
      rerankerReloadBtn: document.getElementById("rerankerReloadBtn"),
      rerankerReloadOut: document.getElementById("rerankerReloadOut"),
    };
    const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
    const tabControls = Array.from(document.querySelectorAll("[data-tab]"));
    const viewLabels = {
      "debug-section": "Embedding 调试台",
      "reranker-section": "Reranker 调试台",
      "results-section": "结果分析",
      "projector-section": "Embedding Projector",
      "admin-section": "运维管理",
    };
    let projectorModulePromise = null;
    let projectorMounted = false;

    async function ensureProjectorMounted() {
      const root = document.getElementById("projector-root");
      if (!root) return;
      try {
        if (!projectorModulePromise) {
          projectorModulePromise = import("/projector-static/projector.js");
        }
        await projectorModulePromise;
        const projectorApi = window.qwenEmbeddingProjector;
        if (!projectorApi) {
          throw new Error("Projector module did not register its browser API.");
        }
        if (!projectorMounted) {
          projectorApi.mountProjector(root);
          projectorMounted = true;
        }
        requestAnimationFrame(() => projectorApi.resizeProjector());
      } catch (error) {
        projectorModulePromise = null;
        root.innerHTML = "";
        const message = document.createElement("div");
        message.className = "projector-loading";
        message.textContent = `Projector 模块加载失败：${error}`;
        root.appendChild(message);
      }
    }

    function setActiveTab(tabId, syncHash = true) {
      const targetId = tabPanels.some(panel => panel.id === tabId) ? tabId : "debug-section";

      tabPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === targetId);
      });
      tabControls.forEach((control) => {
        const isActive = control.dataset.tab === targetId;
        control.classList.toggle("active", isActive);
        control.setAttribute("aria-current", isActive ? "page" : "false");
      });
      els.activeViewLabel.textContent = viewLabels[targetId] || "Qwen3 Embedding & Reranker";

      if (targetId === "projector-section") {
        void ensureProjectorMounted();
      }

      if (syncHash) {
        history.replaceState(null, "", `#${targetId}`);
      }
    }

    function splitTexts(value) {
      return value.split(/\\n+/).map(v => v.trim()).filter(Boolean);
    }

    function showError(message) {
      els.errorBox.classList.add("show");
      els.errorText.textContent = message;
    }

    function clearError() {
      els.errorBox.classList.remove("show");
      els.errorText.textContent = "";
    }

    function setStatus(kind, text) {
      els.statusBox.classList.remove("ok", "warn", "bad");
      els.statusBox.classList.add(kind);
      els.statusText.textContent = text;
      els.topStatusChip.classList.remove("ok", "warn", "bad");
      els.topStatusChip.classList.add(kind);
      els.topStatusChip.textContent = `Backend: ${text}`;
    }

    function cosineSimilarity(a, b) {
      if (!Array.isArray(a) || !Array.isArray(b)) return null;
      let dot = 0;
      let normA = 0;
      let normB = 0;
      const len = Math.min(a.length, b.length);
      for (let i = 0; i < len; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      if (!normA || !normB) return null;
      return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    function renderMatrix(payload) {
      const data = Array.isArray(payload?.data) ? payload.data.slice(0, 8) : [];
      if (!data.length) {
        els.matrixTable.innerHTML = "<tbody><tr><td style='padding:16px;color:var(--muted)'>尚未请求。</td></tr></tbody>";
        return;
      }

      let header = "<thead><tr><th>Text</th>";
      data.forEach((_, idx) => { header += `<th>#${idx + 1}</th>`; });
      header += "</tr></thead>";

      let body = "<tbody>";
      data.forEach((row, rowIndex) => {
        const rowPreview = splitTexts(els.texts.value)[rowIndex] || `item ${rowIndex + 1}`;
        body += `<tr><td title="${rowPreview.replace(/"/g, "&quot;")}">${rowPreview.slice(0, 32)}</td>`;
        data.forEach((col) => {
          const sim = cosineSimilarity(row.embedding, col.embedding);
          body += `<td>${sim === null ? "-" : sim.toFixed(4)}</td>`;
        });
        body += "</tr>";
      });
      body += "</tbody>";
      els.matrixTable.innerHTML = header + body;
    }

    function setSimilarityEmpty(message) {
      els.similarityHeatmap.className = "similarity-empty";
      els.similarityHeatmap.replaceChildren();
      const text = document.createElement("span");
      text.textContent = message;
      els.similarityHeatmap.appendChild(text);
    }

    function renderVectorProfile(vector, container) {
      const sampleCount = Math.min(96, vector.length);
      if (!sampleCount) return;
      const sampled = Array.from({length: sampleCount}, (_, index) => {
        const sourceIndex = sampleCount === 1
          ? 0
          : Math.round((index / (sampleCount - 1)) * (vector.length - 1));
        return Number(vector[sourceIndex]) || 0;
      });
      const maxAbs = Math.max(...sampled.map((value) => Math.abs(value)), 1e-9);
      const points = sampled.map((value, index) => {
        const x = sampleCount === 1 ? 300 : (index / (sampleCount - 1)) * 600;
        const y = 38 - (value / maxAbs) * 27;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");

      const profile = document.createElement("div");
      profile.className = "vector-profile";
      const head = document.createElement("div");
      head.className = "vector-profile-head";
      const title = document.createElement("span");
      title.textContent = "Vector #1 维度轮廓";
      const meta = document.createElement("span");
      meta.textContent = `${sampleCount} 个等距采样点 / ${vector.length} 维`;
      head.append(title, meta);

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 600 76");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "首个 Embedding 向量的维度采样轮廓");
      const baseline = document.createElementNS("http://www.w3.org/2000/svg", "line");
      baseline.setAttribute("x1", "0");
      baseline.setAttribute("x2", "600");
      baseline.setAttribute("y1", "38");
      baseline.setAttribute("y2", "38");
      baseline.setAttribute("stroke", "rgba(148,163,184,.22)");
      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("points", points);
      polyline.setAttribute("fill", "none");
      polyline.setAttribute("stroke", "#ff8a1f");
      polyline.setAttribute("stroke-width", "1.7");
      polyline.setAttribute("vector-effect", "non-scaling-stroke");
      svg.append(baseline, polyline);
      profile.append(head, svg);
      container.appendChild(profile);
    }

    function renderSimilarityVisualization(payload) {
      const data = Array.isArray(payload?.data) ? payload.data.slice(0, 8) : [];
      if (!data.length) {
        setSimilarityEmpty("当前没有可视化的向量结果。");
        return;
      }
      const vectors = data.map((item) => item.embedding);
      if (!vectors.every(Array.isArray)) {
        setSimilarityEmpty("base64 编码结果无法直接计算相似度；请将 Encoding Format 切换为 float。");
        return;
      }

      const texts = splitTexts(els.texts.value);
      els.similarityHeatmap.className = "";
      els.similarityHeatmap.replaceChildren();

      const wrap = document.createElement("div");
      wrap.className = "similarity-grid-wrap";
      const grid = document.createElement("div");
      grid.className = "similarity-grid";
      grid.style.gridTemplateColumns = `minmax(120px, 1.4fr) repeat(${data.length}, minmax(52px, 1fr))`;
      grid.style.minWidth = `${120 + data.length * 52}px`;

      const corner = document.createElement("div");
      corner.className = "similarity-axis row-label";
      corner.textContent = `前 ${data.length} 条文本`;
      grid.appendChild(corner);
      data.forEach((_, index) => {
        const header = document.createElement("div");
        header.className = "similarity-axis";
        header.textContent = `#${index + 1}`;
        grid.appendChild(header);
      });

      data.forEach((row, rowIndex) => {
        const rowLabel = document.createElement("div");
        rowLabel.className = "similarity-axis row-label";
        const rowText = texts[rowIndex] || `item ${rowIndex + 1}`;
        rowLabel.textContent = `#${rowIndex + 1} ${rowText}`;
        rowLabel.title = rowText;
        grid.appendChild(rowLabel);

        data.forEach((col, colIndex) => {
          const similarity = cosineSimilarity(row.embedding, col.embedding);
          const cell = document.createElement("div");
          cell.className = "similarity-cell";
          if (similarity === null || !Number.isFinite(similarity)) {
            cell.textContent = "-";
            cell.style.setProperty("--cell-alpha", ".06");
          } else {
            const strength = Math.min(1, Math.abs(similarity));
            cell.textContent = similarity.toFixed(3);
            cell.style.setProperty("--cell-alpha", (0.08 + strength * 0.62).toFixed(3));
            cell.classList.toggle("negative", similarity < 0);
            cell.title = `#${rowIndex + 1} ↔ #${colIndex + 1}: ${similarity.toFixed(6)}`;
          }
          grid.appendChild(cell);
        });
      });

      wrap.appendChild(grid);
      const legend = document.createElement("div");
      legend.className = "similarity-legend";
      const low = document.createElement("span");
      low.textContent = "低相似 / 负相关";
      const gradient = document.createElement("span");
      gradient.className = "similarity-gradient";
      const high = document.createElement("span");
      high.textContent = "高相似";
      legend.append(low, gradient, high);
      els.similarityHeatmap.append(wrap, legend);
      renderVectorProfile(vectors[0], els.similarityHeatmap);
    }

    function summarizeResponse(payload, latencyMs) {
      const data = Array.isArray(payload.data) ? payload.data : [];
      const first = data[0]?.embedding || [];
      const lines = [];
      lines.push(`vectors: ${data.length}`);
      lines.push(`dimensions: ${first.length || "-"}`);
      lines.push(`latency_ms: ${latencyMs}`);
      lines.push(`model: ${payload.model || "-"}`);
      if (payload.usage) {
        lines.push(`usage.prompt_tokens: ${payload.usage.prompt_tokens ?? "-"}`);
        lines.push(`usage.total_tokens: ${payload.usage.total_tokens ?? "-"}`);
      }
      lines.push("");
      data.forEach((item, idx) => {
        const preview = Array.isArray(item.embedding) ? item.embedding.slice(0, 12) : [];
        lines.push(`embedding[${idx}] head: ${JSON.stringify(preview)}`);
      });
      return lines.join("\\n");
    }

    function buildPayload() {
      const texts = splitTexts(els.texts.value);
      if (!texts.length) return null;
      const payload = {input: texts.length === 1 ? texts[0] : texts};
      if (els.inputType.value) payload.input_type = els.inputType.value;
      if (els.instruction.value.trim()) payload.instruction = els.instruction.value.trim();
      if (els.dimensions.value.trim()) payload.dimensions = Number(els.dimensions.value);
      if (els.encodingFormat.value) payload.encoding_format = els.encodingFormat.value;
      return payload;
    }

    function applyTemplate(name) {
      const selected = templates[name];
      if (!selected) return;
      els.templateChip.textContent = `Template: ${selected.label}`;
      els.templateHelp.innerHTML = `<div class="template-note">${selected.note}</div>`;
      if (name === "blank") return;
      els.texts.value = selected.texts;
      els.inputType.value = selected.inputType;
      els.instruction.value = selected.instruction;
      els.dimensions.value = selected.dimensions;
      els.encodingFormat.value = selected.encodingFormat;
    }

    function copyJson() {
      if (!lastEmbeddingPayload) {
        showError("当前还没有可复制的 JSON 结果。");
        return;
      }
      navigator.clipboard.writeText(JSON.stringify(lastEmbeddingPayload, null, 2))
        .then(() => setStatus("ok", "JSON copied to clipboard"))
        .catch((error) => showError(`复制失败：${error}`));
    }

    function downloadJson() {
      if (!lastEmbeddingPayload) {
        showError("当前还没有可下载的结果。");
        return;
      }
      const blob = new Blob([JSON.stringify(lastEmbeddingPayload, null, 2)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "qwen3-embedding-response.json";
      anchor.click();
      URL.revokeObjectURL(url);
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health");
        const payload = await response.json();
        lastHealthPayload = payload;
        els.healthOut.textContent = JSON.stringify(payload, null, 2);
        els.healthModelChip.textContent = payload.model_id || "-";
        els.healthDeviceChip.textContent = payload.backend_target_device || "cuda";
        els.topModelChip.textContent = `Model: ${payload.model_id || "-"}`;
        els.topDeviceChip.textContent = `Target: ${payload.backend_target_device || "cuda"}`;
        els.sidebarModel.textContent = payload.model_id || "-";
        els.sidebarDevice.textContent = payload.backend_target_device || "cuda";
        els.sidebarDtype.textContent = payload.dtype || "-";
        els.metricState.textContent = payload.backend_state || "-";
        if (Number.isInteger(payload.max_dimensions) && payload.max_dimensions > 0) {
          els.dimensions.max = String(payload.max_dimensions);
          els.dimensions.title = `当前模型最大输出维度：${payload.max_dimensions}`;
        } else {
          els.dimensions.removeAttribute("max");
          els.dimensions.title = "输出维度上限会从当前模型配置自动读取";
        }
        const reranker = payload.reranker || {};
        if (reranker.quantization) els.rerankerQuantization.value = reranker.quantization;
        const message = payload.backend_last_error || reranker.last_error || `Backend ${payload.backend_state || "unknown"}`;
        if (payload.backend_ready && reranker.ready) {
          clearError();
          setStatus("ok", "embedding + reranker ready");
        } else if (payload.backend_ready) {
          showError(message || "Reranker 尚未就绪；Embedding 仍可使用。");
          setStatus("warn", `embedding ready / reranker ${reranker.state || "not ready"}`);
        } else if (payload.backend_state === "starting") {
          showError(message);
          setStatus("warn", "starting / loading");
        } else {
          showError(message);
          setStatus("bad", payload.backend_state || "not ready");
        }
      } catch (error) {
        showError(`Health check failed: ${error}`);
        setStatus("bad", "health check failed");
      }
    }

    async function runEmbedding() {
      const payload = buildPayload();
      if (!payload) {
        showError("请至少输入一条文本。");
        setStatus("bad", "missing input");
        return;
      }

      if (!lastHealthPayload || !lastHealthPayload.backend_ready) {
        await refreshHealth();
        if (!lastHealthPayload || !lastHealthPayload.backend_ready) {
          showError(
            (lastHealthPayload && lastHealthPayload.backend_last_error)
            || "Backend 尚未 ready。若模型仍在加载 GPU，请等待片刻后重试。"
          );
          return;
        }
      }

      clearError();
      els.runBtn.disabled = true;
      setStatus("warn", "requesting");
      setSimilarityEmpty("正在生成并计算向量关系...");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const startedAt = performance.now();

      try {
        const response = await fetch("/v1/embeddings", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        const result = await response.json();
        const latencyMs = Math.round(performance.now() - startedAt);
        if (!response.ok) {
          const backendMessage = result?.error?.message || `HTTP ${response.status}`;
          if (backendMessage.includes("does not support matryoshka")) {
            els.dimensions.value = "";
            showError(`${backendMessage} 当前模型不支持自定义 dimensions，已自动清空该字段，请重试。`);
            setStatus("warn", "dimensions disabled");
          } else {
            showError(backendMessage);
            setStatus("bad", "request failed");
          }
          els.rawOut.textContent = JSON.stringify(result, null, 2);
          els.summaryOut.textContent = "请求失败。";
          renderMatrix(null);
          setSimilarityEmpty("请求失败，暂无可视化结果。");
          return;
        }

        lastEmbeddingPayload = result;
        const vectorCount = Array.isArray(result.data) ? result.data.length : 0;
        const vectorDim = result.data?.[0]?.embedding?.length || "-";
        els.metricCount.textContent = String(vectorCount);
        els.metricDim.textContent = String(vectorDim);
        els.metricLatency.textContent = `${latencyMs} ms`;
        els.summaryOut.textContent = summarizeResponse(result, latencyMs);
        els.rawOut.textContent = JSON.stringify(result, null, 2);
        renderMatrix(result);
        renderSimilarityVisualization(result);
        setStatus("ok", "embedding completed");
        await refreshHealth();
      } catch (error) {
        if (error && error.name === "AbortError") {
          showError("请求超时：backend 可能仍在加载模型，或后端响应时间过长。");
          setStatus("warn", "request timeout");
        } else {
          showError(`Request failed: ${error}`);
          setStatus("bad", "request failed");
        }
        setSimilarityEmpty("请求失败，暂无可视化结果。");
      } finally {
        clearTimeout(timer);
        els.runBtn.disabled = false;
      }
    }

    async function submitReload() {
      const payload = {};
      if (els.reloadModelId.value.trim()) payload.model_id = els.reloadModelId.value.trim();
      if (els.reloadDtype.value.trim()) payload.dtype = els.reloadDtype.value.trim();
      if (els.reloadMaxModelLen.value.trim()) payload.max_model_len = Number(els.reloadMaxModelLen.value);
      if (els.reloadGpuUtil.value.trim()) payload.gpu_memory_utilization = Number(els.reloadGpuUtil.value);
      if (els.reloadInstruction.value.trim()) payload.default_query_instruction = els.reloadInstruction.value.trim();
      if (els.reloadExtraArgs.value.trim()) payload.extra_args = els.reloadExtraArgs.value.trim();

      els.reloadBtn.disabled = true;
      els.reloadOut.textContent = "提交中...";
      try {
        const response = await fetch("/admin/reload", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-admin-token": els.reloadToken.value.trim(),
          },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        els.reloadOut.textContent = JSON.stringify(result, null, 2);
        if (!response.ok) {
          showError(result?.error?.message || `Reload failed with HTTP ${response.status}`);
          setStatus("bad", "reload failed");
          return;
        }
        clearError();
        setStatus("warn", "reloading backend");
        await refreshHealth();
      } catch (error) {
        els.reloadOut.textContent = String(error);
        showError(`Reload request failed: ${error}`);
      } finally {
        els.reloadBtn.disabled = false;
      }
    }

    function renderRerankResults(payload, latencyMs) {
      const results = Array.isArray(payload?.results) ? payload.results : [];
      els.rerankResults.replaceChildren();
      results.forEach((item, rank) => {
        const card = document.createElement("div");
        card.className = "rerank-result";
        const head = document.createElement("div");
        head.className = "rerank-result-head";
        const position = document.createElement("span");
        position.textContent = `#${rank + 1} · 原始索引 ${item.index}`;
        const score = document.createElement("strong");
        score.textContent = `score ${Number(item.relevance_score).toFixed(6)}`;
        head.append(position, score);
        const text = document.createElement("div");
        text.className = "rerank-result-text";
        text.textContent = item?.document?.text || "";
        card.append(head, text);
        els.rerankResults.appendChild(card);
      });
      els.rerankMeta.textContent = `${results.length} 条结果 · ${latencyMs} ms · ${payload?.model || "-"}`;
      els.rerankRawOut.textContent = JSON.stringify(payload, null, 2);
    }

    async function runRerank() {
      const query = els.rerankQuery.value.trim();
      const documents = splitTexts(els.rerankDocuments.value);
      if (!query || !documents.length) {
        showError("Reranker 需要非空 query 和至少一篇文档。");
        return;
      }
      const payload = {query, documents};
      if (els.rerankTopN.value.trim()) payload.top_n = Number(els.rerankTopN.value);
      els.rerankBtn.disabled = true;
      clearError();
      setStatus("warn", "reranking");
      const startedAt = performance.now();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      try {
        const response = await fetch("/v1/rerank", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        const result = await response.json();
        if (!response.ok) {
          els.rerankRawOut.textContent = JSON.stringify(result, null, 2);
          showError(result?.error?.message || `Rerank failed with HTTP ${response.status}`);
          setStatus("bad", "rerank failed");
          return;
        }
        lastRerankPayload = result;
        renderRerankResults(result, Math.round(performance.now() - startedAt));
        setStatus("ok", "rerank completed");
      } catch (error) {
        showError(error?.name === "AbortError" ? "Rerank 请求超时。" : `Rerank 请求失败：${error}`);
        setStatus("bad", "rerank failed");
      } finally {
        clearTimeout(timer);
        els.rerankBtn.disabled = false;
      }
    }

    async function submitRerankerReload() {
      const payload = {quantization: els.rerankerQuantization.value};
      if (els.rerankerRevision.value.trim()) payload.model_revision = els.rerankerRevision.value.trim();
      els.rerankerReloadBtn.disabled = true;
      els.rerankerReloadOut.textContent = "提交中...";
      try {
        const response = await fetch("/admin/reranker/reload", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-admin-token": els.reloadToken.value.trim(),
          },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        els.rerankerReloadOut.textContent = JSON.stringify(result, null, 2);
        if (!response.ok) {
          showError(result?.error?.message || `Reranker reload failed with HTTP ${response.status}`);
          return;
        }
        clearError();
        await refreshHealth();
      } catch (error) {
        els.rerankerReloadOut.textContent = String(error);
        showError(`Reranker reload 请求失败：${error}`);
      } finally {
        els.rerankerReloadBtn.disabled = false;
      }
    }

    els.applyTemplateBtn.addEventListener("click", () => applyTemplate(els.templateSelect.value));
    els.fillDemoBtn.addEventListener("click", () => applyTemplate("retrieval_pair"));
    els.healthBtn.addEventListener("click", refreshHealth);
    els.runBtn.addEventListener("click", runEmbedding);
    els.copyJsonBtn.addEventListener("click", copyJson);
    els.downloadBtn.addEventListener("click", downloadJson);
    els.reloadBtn.addEventListener("click", submitReload);
    els.rerankBtn.addEventListener("click", runRerank);
    els.rerankerReloadBtn.addEventListener("click", submitRerankerReload);
    tabControls.forEach((control) => {
      control.addEventListener("click", () => {
        setActiveTab(control.dataset.tab || "debug-section");
      });
    });
    window.addEventListener("hashchange", () => {
      setActiveTab(window.location.hash.replace("#", "") || "debug-section", false);
    });

    setActiveTab(window.location.hash.replace("#", "") || "debug-section", false);
    applyTemplate("retrieval_pair");
    refreshHealth();
  </script>
</body>
</html>
"""
    return (
        template.replace("__MODEL_ID__", get_current_model_id())
        .replace("__DEFAULT_INSTRUCTION__", DEFAULT_QUERY_INSTRUCTION)
        .replace("__RERANKER_MODEL_ID__", get_current_reranker_model_id())
    )


def create_application() -> FastAPI:
    mcp = create_mcp_server()
    app_root = os.path.dirname(__file__)
    projector_dist_dir = os.path.join(app_root, "static", "projector")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            preload_task: Optional[asyncio.Task] = None
            preload_task = asyncio.create_task(_preload_backends())
            try:
                yield
            finally:
                if preload_task is not None and not preload_task.done():
                    preload_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await preload_task
                await shutdown_reranker()
                await shutdown_backend()

    app = FastAPI(title="Qwen3 Embedding & Reranker", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_, exc: RequestValidationError) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", [])[1:])
        message = first_error.get("msg", "Invalid request body.")
        if location:
            message = f"{location}: {message}"
        return _error_response(message, 400, "invalid_request_error", 400)
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(app_root, "static")),
        name="static",
    )
    if os.path.isdir(projector_dist_dir):
        app.mount("/projector-static", StaticFiles(directory=projector_dist_dir), name="projector-static")
    app.mount("/mcp", mcp.streamable_http_app(), name="mcp")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _build_index_html()

    @app.get("/projector", response_class=RedirectResponse)
    def projector_index() -> RedirectResponse:
        return RedirectResponse(url="/#projector-section", status_code=307)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(await get_health_payload())

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> JSONResponse:
        try:
            response_payload = await create_embeddings(request.model_dump(exclude_none=True))
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        except BackendProxyError as exc:
            if exc.payload is not None:
                return JSONResponse(exc.payload, status_code=exc.status_code)
            return _error_response(str(exc), exc.status_code, "backend_error", exc.status_code)
        return JSONResponse(response_payload)

    @app.post("/v1/rerank")
    async def rerank(request: RerankRequest) -> JSONResponse:
        try:
            response_payload = await create_rerank(request.model_dump(exclude_none=True))
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        except BackendProxyError as exc:
            if exc.payload is not None:
                return JSONResponse(exc.payload, status_code=exc.status_code)
            return _error_response(str(exc), exc.status_code, "backend_error", exc.status_code)
        return JSONResponse(response_payload)

    @app.post("/v1/embeddings/projector")
    async def projector_embeddings(request: ProjectorRequest) -> JSONResponse:
        try:
            response_payload = await create_projector_payload(
                request.model_dump(exclude_none=True),
                embedder=create_embeddings,
            )
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        except BackendProxyError as exc:
            if exc.payload is not None:
                return JSONResponse(exc.payload, status_code=exc.status_code)
            return _error_response(str(exc), exc.status_code, "backend_error", exc.status_code)
        except ProjectorDependencyError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        return JSONResponse(response_payload)

    @app.post("/admin/reload")
    async def admin_reload(request: ReloadRequest, x_admin_token: Optional[str] = Header(default=None)) -> JSONResponse:
        try:
            _authorize_admin(x_admin_token)
            payload = await reload_backend(request.model_dump(exclude_none=True))
        except PermissionError as exc:
            return _error_response(str(exc), 401, "unauthorized", 401)
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        return JSONResponse(payload)

    @app.post("/admin/reranker/reload")
    async def admin_reranker_reload(
        request: RerankerReloadRequest,
        x_admin_token: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        try:
            _authorize_admin(x_admin_token)
            payload = await reload_reranker(request.model_dump(exclude_none=True))
        except PermissionError as exc:
            return _error_response(str(exc), 401, "unauthorized", 401)
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        return JSONResponse(payload)

    @app.get("/openai-example", response_class=HTMLResponse)
    def openai_example() -> str:
        example = {
            "base_url": "http://localhost:12302/v1",
            "model": get_current_model_id(),
            "input": "hello world",
            "input_type": "query",
            "instruction": DEFAULT_QUERY_INSTRUCTION,
            "reranker_model": get_current_reranker_model_id(),
            "rerank_endpoint": "http://localhost:12302/v1/rerank",
        }
        return "<pre>" + json.dumps(example, ensure_ascii=False, indent=2) + "</pre>"

    return app


app = create_application()
