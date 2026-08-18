FROM node:20-bookworm-slim AS projector-builder

WORKDIR /frontend

COPY frontend/package.json /frontend/package.json
RUN npm install --no-audit --no-fund
COPY frontend /frontend
RUN npm run build


FROM vllm/vllm-openai:v0.19.1

ENV TZ=Asia/Shanghai

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /app/requirements.txt
# vLLM 0.19.1 ships an unused system PyGObject package without its optional
# PyCairo dependency. This service has no GTK/GObject code, so remove that
# orphan before enforcing a clean Python dependency graph.
RUN pip uninstall --yes pygobject \
    && pip check

COPY app.py /app/app.py
COPY embedding_service.py /app/embedding_service.py
COPY reranker_service.py /app/reranker_service.py
COPY service_health.py /app/service_health.py
COPY process_utils.py /app/process_utils.py
COPY projector_service.py /app/projector_service.py
COPY mcp_server.py /app/mcp_server.py
COPY server.py /app/server.py
COPY templates /app/templates
COPY scripts /app/scripts
COPY static /app/static
COPY --from=projector-builder /frontend/dist /app/static/projector

EXPOSE 12302
ENTRYPOINT ["python3", "-u", "server.py"]
