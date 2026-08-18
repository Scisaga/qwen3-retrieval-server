# Qwen3 Retrieval

自托管的 Qwen3 检索推理服务。在一个应用容器内管理两个相互独立的 vLLM 0.19.1 后端：`Qwen/Qwen3-Embedding-4B`
负责第一阶段向量召回，`Qwen/Qwen3-Reranker-0.6B` 负责第二阶段精排。默认两者只使用
GPU 0，公共入口仍为 `:12302`；项目不包含向量库、切块或生成编排。

项目地址：
- 代码仓库：[`https://github.com/Scisaga/qwen3-retrieval-server`](https://github.com/Scisaga/qwen3-retrieval-server)
- 镜像仓库（GHCR）：`ghcr.io/scisaga/qwen3-retrieval-server:latest`
- 兼容镜像名：`ghcr.io/scisaga/qwen3-embedding-openai:latest`（迁移期保留）

## 功能
- OpenAI 兼容 Embeddings API：`POST /v1/embeddings`
- vLLM 兼容 Rerank API：`POST /v1/rerank`（1–50 篇纯文本文档）
- 3D Projector API：`POST /v1/embeddings/projector`（后端预计算 3D 投影 + 近邻）
- Qwen 检索增强字段：`input_type=query|document`、`instruction`
- MCP Server：HTTP 挂载到 `POST/GET /mcp`，提供 embedding、rerank 与 projector 工具
- 内置 Web UI：`GET /`，按“工作台 / 分析工具 / 系统”组织 Embedding、Reranker、向量投影和服务管理
- Embedding 内部结果分析：直接展示余弦相似度热力图、首个向量的维度采样轮廓与原始响应
- 向量投影：`GET /#projector-section`（3D 点云、原点连线、箭头、坐标轴、近邻联动）；`GET /projector` 保留为兼容跳转
- 交互式接口文档：`GET /docs`（Swagger UI）与 `GET /redoc`
- 模型自动下载与缓存：将 `./models` 挂载到容器 `/models`（Hugging Face 缓存目录）
- 独立生命周期：先加载 Embedding、再加载 Reranker；Reranker 失败时 Embedding 继续服务，聚合健康为 `degraded`
- 运维友好：`GET /health` 保留 Embedding 顶层字段，并增加 `reranker`；两个后端可独立热重载
- GitHub Actions：自动构建并发布 Docker 镜像到 GHCR（`.github/workflows/docker-publish.yml`）

## 快速开始
```bash
docker compose up -d --build
```

说明：Embedding 使用 `--runner pooling --convert embed` 和 Matryoshka override；Reranker 使用
`--runner pooling --convert classify`、项目内固化的官方 Qwen3 模板和固定检索 instruction。

如果机器需要走代理才能访问 Hugging Face，可在同目录创建 `.env`（或启动前导出环境变量）：

```bash
HTTP_PROXY=http://127.0.0.1:7890
# 可选：不走代理的地址（默认：localhost,127.0.0.1）
# NO_PROXY=localhost,127.0.0.1
```

打开：
- Web UI：http://localhost:12302/
- Projector：http://localhost:12302/#projector-section
- MCP HTTP：http://localhost:12302/mcp
- 接口文档（Swagger）：http://localhost:12302/docs
- 接口文档（ReDoc）：http://localhost:12302/redoc
- 健康检查：http://localhost:12302/health

## OpenAI SDK 快速开始

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12302/v1",
    api_key="dummy",
)

response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-4B",
    input="What is the capital of China?",
    extra_body={
        "input_type": "query",
        "instruction": "Given a web search query, retrieve relevant passages that answer the query",
        "dimensions": 1024,
    },
)

print(len(response.data[0].embedding))
```

Rerank 尚不是 OpenAI SDK 的标准资源，可直接使用 `httpx`：

```python
import httpx

response = httpx.post(
    "http://localhost:12302/v1/rerank",
    json={
        "query": "中国的首都是哪里？",
        "documents": ["巴黎是法国首都。", "北京是中国首都。"],
        "top_n": 1,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["results"][0])
```

## curl 示例

### Embeddings
```bash
curl http://localhost:12302/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": [
      "What is the capital of China?",
      "The capital of China is Beijing."
    ],
    "input_type": "query",
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "dimensions": 1024
  }'
```

### Reranker

```bash
curl http://localhost:12302/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of China?",
    "documents": [
      "Paris is the capital of France.",
      "Beijing is the capital of China.",
      "Python uses indentation for blocks."
    ],
    "top_n": 2
  }'
```

响应按 `relevance_score` 降序排列，每项保留输入数组的原始 `index` 和 `document.text`。
省略 `top_n` 时返回全部结果。请求级 instruction、模板、优先级与截断参数均不开放。

### Projector
```bash
curl http://localhost:12302/v1/embeddings/projector \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      "What is the capital of China?",
      "The capital of China is Beijing.",
      "Paris is the capital of France."
    ],
    "labels": ["query", "fact", "fact"],
    "projection_method": "umap",
    "metric": "cosine",
    "neighbors_k": 10,
    "point_size": 5
  }'
```

## MCP 快速开始

### HTTP MCP
服务启动后，MCP Streamable HTTP 入口固定为：

```text
http://localhost:12302/mcp
```

适合远端客户端或通过网关统一接入的场景。

## MCP 能力一览

### Tools
- `embed_text`
  - 入参：`texts`（必填，字符串或字符串数组）、`input_type`（可选）、`instruction`（可选）、`dimensions`（可选）
  - 返回：标准 OpenAI embeddings 响应形状
- `project_texts`
  - 入参：`texts`（必填）、`labels`（可选）、`projection_method`（可选，`umap|tsne|pca`）、`metric`（可选，`cosine|euclidean`）、`neighbors_k`（可选）、`point_size`（可选）
  - 返回：Projector 负载（`points`、`neighbors`、`projection_meta`）
- `rerank_documents`
  - 入参：`query`、`documents`（1–50 个非空字符串）、可选 `top_n`
  - 返回：按分数降序的 vLLM Rerank 响应

### Resources
- `qwen3retrieval://health`：Embedding 顶层状态与嵌套 Reranker 状态
- `qwen3retrieval://usage`：MCP 工具参数说明与使用建议
- `qwen3embedding://health` / `qwen3embedding://usage`：项目改名前的兼容别名

### Prompts
- `retrieval_embedding_workflow`：指导客户端如何区分 query/document，并在 query 侧传入 instruction
- `rag_retrieval_workflow`：指导客户端先向量召回，再把不超过 50 条候选交给 Reranker
- `projector_workflow`：指导客户端如何构建可视化聚类与近邻探索请求

## 架构说明
- **对外端口**：`PORT=12302`
- **Embedding vLLM**：`127.0.0.1:8001`
- **Reranker vLLM**：`127.0.0.1:8002`
- **启动/关闭**：按 Embedding → Reranker 启动，按 Reranker → Embedding 关闭；Reranker 重载不影响 Embedding
- **GPU**：Compose 只暴露 `device_ids: ["0"]`，GPU 1 完全不可见
- **自动下载**：首次启动会把两个模型缓存到挂载的 `/models`

这意味着你通常只需要访问：

```text
http://localhost:12302/v1/embeddings
http://localhost:12302/v1/rerank
```

无需直接访问容器内 `8001`。

## 接口一览
- `POST /v1/embeddings`
  - 标准字段：`input`、`model`、`dimensions`、`encoding_format`、`user`
  - 扩展字段：`input_type`、`instruction`
- `POST /v1/embeddings/projector`
  - 字段：`inputs`、`labels`（可选）、`input_type`（可选）、`instruction`（可选）
  - 投影参数：`projection_method=umap|tsne|pca`、`metric=cosine|euclidean`、`neighbors_k`、`point_size`
  - 返回：`points`（3D 坐标 + 文本元数据）、`neighbors`、`projection_meta`、`usage`
- `POST /v1/rerank`
  - `query`：非空字符串
  - `documents`：1–50 个非空字符串
  - `top_n`：可选，范围 `1..len(documents)`；省略时返回全部
  - `model` / `user`：可选；`model` 若提供必须等于当前服务模型
  - 禁止额外字段；无效输入返回统一 400，未就绪返回 503，后端错误原样透传
- `POST /mcp` / `GET /mcp`：MCP Streamable HTTP 入口
- `GET /`：统一 Web 控制台；`GET /projector` 会跳转到其中的 Projector 页签
- `GET /docs` / `GET /redoc`：交互式接口文档
- `GET /openapi.json`：OpenAPI 规范 JSON
- `GET /health`：健康检查与运行参数
- `POST /admin/reload`：热重载模型（需 `x-admin-token`）
- `POST /admin/reranker/reload`：独立重载 Reranker，仅接受 `model_revision` 和 `quantization=none|bitsandbytes`

## GitHub Workflow：自动构建发布镜像

仓库内置工作流：`.github/workflows/docker-publish.yml`

- 触发条件：
  - push 到 `main`
  - push `v*` 标签
  - `pull_request` 到 `main`（仅构建，不推送）
  - 手动触发 `workflow_dispatch`
- 主镜像仓库：`ghcr.io/<owner>/qwen3-retrieval-server`
- 迁移期同时发布旧镜像名 `ghcr.io/<owner>/qwen3-embedding-openai`
- 标签策略：`latest`（默认分支）、分支名、tag 名、commit sha

首次使用时请确保：

1. 仓库已开启 GitHub Packages（GHCR）权限
2. Actions 具备 `packages: write`（工作流已声明）
3. 如果仓库是组织仓库，组织策略允许 `GITHUB_TOKEN` 推送包

## Docker 部署示例
```bash
docker run -d --name qwen3_retrieval_server \
  --gpus '"device=0"' \
  -p 12302:12302 \
  -e MODEL_ID="Qwen/Qwen3-Embedding-4B" \
  -e HF_HOME="/models" \
  -v ./models:/models \
  ghcr.io/scisaga/qwen3-retrieval-server:latest
```

如果你绕过本仓库、直接调用原生 `vllm serve` 启动 `Qwen3-Embedding-*`，请显式追加：

```bash
--hf-overrides '{"is_matryoshka": true}'
```

否则某些 vLLM 版本会把 `dimensions=1024` 这类请求误判为不支持，返回 HTTP 400。

## 切换模型（需重启）
在 `docker-compose.yml` 中修改 `MODEL_ID`，然后：

```bash
docker compose up -d
```

## 模型热重载（无需重启）
```bash
curl -X POST http://localhost:12302/admin/reload \
  -H "Content-Type: application/json" \
  -H "x-admin-token: change-me" \
  -d '{
    "model_id":"Qwen/Qwen3-Embedding-4B",
    "max_model_len":4096,
    "gpu_memory_utilization":0.72
  }'
```

独立重载 Reranker：

```bash
curl -X POST http://localhost:12302/admin/reranker/reload \
  -H "Content-Type: application/json" \
  -H "x-admin-token: change-me" \
  -d '{"quantization":"none"}'
```

该接口不能在线提高显存比例、上下文长度或并发上限。

## GPU 绑定

本项目的发布配置固定为单卡：Compose `device_ids: ["0"]` 且
`AUTO_BACKEND_REPLICAS=0`。不要把 GPU 1 加入可见设备，也不要启用 tensor/pipeline parallel。
容器内看到的 `cuda:0` 对应宿主机 GPU 0。

## Projector 说明
- 前端采用 `Vite + Plotly`（目录：`frontend/`）
- Docker 构建会自动打包前端并复制到 `static/projector`；Projector 模块在首次打开主页对应页签时按需加载
- 旧地址 `/projector` 会临时跳转到 `/#projector-section`
- 当前可视化为 3D 点云，包含：
  - 点编号标签
  - 原点及原点到各点连线与箭头
  - 三维坐标轴（X 红 / Y 绿 / Z 蓝）
  - 点击点后在下方分析卡片中展示近邻

本地前端开发：

```bash
cd frontend
npm install
npm run dev
```

本地前端打包：

```bash
cd frontend
npm install
npm run build
```

## 常用环境变量
在 `docker-compose.yml` 的 `environment` 里可调：
- `MODEL_ID`：模型 ID，默认 `Qwen/Qwen3-Embedding-4B`
- `PORT`：外层 FastAPI 端口，默认 `12302`
- `BACKEND_HOST` / `BACKEND_PORT`：容器内 vLLM 监听地址
- `HF_HOME`：模型缓存目录
- `DTYPE`：模型精度，默认 `float16`
- `MAX_MODEL_LEN`：最大上下文长度，默认 `4096`
- `GPU_MEMORY_UTILIZATION`：Embedding vLLM 显存预留比例；Compose 为保留 8192 上下文并与同卡 ASR/Reranker 共存固定为 `0.47`
- `DEFAULT_QUERY_INSTRUCTION`：query 侧默认 instruction
- `ADMIN_TOKEN`：热重载接口鉴权
- `VLLM_EXTRA_ARGS`：透传额外 vLLM 参数
- `AUTO_BACKEND_REPLICAS`：多卡默认按“每张卡 1 个 vLLM 实例”启动，设为 `0` 可关闭
- `BACKEND_REPLICA_COUNT`：显式指定启动多少个单卡 vLLM 副本
- `PROJECTOR_CACHE_TTL_SECONDS`：Projector 结果缓存 TTL（秒）
- `PROJECTOR_CACHE_MAX_ITEMS`：Projector 缓存项上限
- `RERANKER_MODEL_ID`：默认 `Qwen/Qwen3-Reranker-0.6B`
- `RERANKER_MODEL_REVISION`：可选模型 revision
- `RERANKER_BACKEND_PORT`：固定默认 `8002`
- `RERANKER_DTYPE`：固定发布值 `float16`
- `RERANKER_MAX_MODEL_LEN`：固定发布值 `2048`
- `RERANKER_GPU_MEMORY_UTILIZATION`：FP16 初始值 `0.08`
- `RERANKER_FALLBACK_GPU_MEMORY_UTILIZATION`：仅 FP16 启动失败时尝试 `0.085`
- `RERANKER_QUANTIZED_GPU_MEMORY_UTILIZATION`：BitsAndBytes 4-bit 使用 `0.06`，保留 2048-token 上下文和单序列调度
- `RERANKER_QUANTIZATION`：`none` 或 `bitsandbytes`
- `RERANKER_PRELOAD_RETRY_DELAY`：冷启动 profiling 竞态后的同配置重试等待秒数，Compose 为 `5`

注意：
- 输出向量维度上限无需配置：服务会读取当前模型的 `config.json` 自动确定，并通过 `/health` 的 `max_dimensions` 返回（4B 为 `2560`）。热重载模型时也会重新解析。
- 如果手动设置 `--max-num-batched-tokens`，它不能小于 `MAX_MODEL_LEN`；否则 vLLM 会在启动阶段报错退出。
- 不建议使用 `VLLM_PORT` 作为 wrapper 配置名。该变量会与 vLLM 内部端口逻辑冲突，可能造成误导日志。

## 测试
```bash
python -m pytest
python -m compileall -q app.py embedding_service.py reranker_service.py service_health.py mcp_server.py projector_service.py
cd frontend && npm install && npm run build
docker compose config --quiet
docker build -t qwen3-retrieval-server:reranker-candidate .
```

## 显存与量化验收

Reranker 的 2 GiB 门槛定义为：根 PID 及其全部 GPU 子进程在 NVML 中的
`used_memory` 之和，100 ms 采样，所有场景峰值必须严格小于 2048 MiB。脚本应在 Docker
宿主机执行，它会覆盖空载、单文档、50×约 512-token、接近 2048-token 和四个并发客户端：

```bash
python scripts/measure_reranker_vram.py \
  --container qwen3_retrieval_server \
  --base-url http://127.0.0.1:12302 \
  --gpu-index 0
```

仅当 FP16 无法满足启动或显存门槛时才考虑 BitsAndBytes。先记录 FP16 基线，再切换候选并比较：

```bash
python scripts/evaluate_reranker_quantization.py \
  --record-fp16-url http://127.0.0.1:12302 \
  --baseline-file /tmp/reranker-fp16.json

python scripts/evaluate_reranker_quantization.py \
  --candidate-url http://127.0.0.1:12302 \
  --baseline-file /tmp/reranker-fp16.json
```

门槛为 Top-1 一致率至少 95%，nDCG@10 与 MRR@10 的绝对下降均不超过 `0.0121`。

当前 GPU 0 实机候选结果（2026-08-18）：

- FP16 / `gpu_memory_utilization=0.08`：可启动，NVML 进程树五场景峰值 `2414 MiB`，显存门禁失败。
- BitsAndBytes 4-bit / `gpu_memory_utilization=0.06`：峰值 `1976 MiB`，显存门禁通过。
- 4-bit 对 FP16 的 Top-1 一致率 `95%`、MRR@10 下降 `0`，nDCG@10 绝对下降 `0.0120702`，低于 `0.0121` 上限。

因此 BitsAndBytes 4-bit 同时通过显存与质量门禁，Compose 发布配置固定为 4-bit / `0.06`。

## License
MIT License
