import Plotly from "plotly.js-dist-min";
import "./styles.css";

let mountedProjector = null;

export function resizeProjector() {
  if (!mountedProjector?.plot?.isConnected) return;
  requestAnimationFrame(() => {
    try {
      Plotly.Plots.resize(mountedProjector.plot);
    } catch (_) {
      // Plotly has not rendered a scene yet.
    }
  });
}

export function mountProjector(root) {
  if (!root) throw new Error("Projector root element is required.");
  if (root.dataset.projectorMounted === "true") {
    resizeProjector();
    return;
  }
  const app = root;

app.innerHTML = `
  <div class="projector-view">
    <div class="projector-grid">
      <section class="projector-card projector-controls">
        <div class="projector-card-header">
          <div>
            <div class="projector-card-title">Projector Controls</div>
            <div class="projector-card-subtitle">POST <code>/v1/embeddings/projector</code></div>
          </div>
        </div>
        <div class="projector-card-body projector-form">
          <label class="projector-field" for="projectorInputs">
            <span class="projector-label">Texts（每行一条）</span>
            <textarea id="projectorInputs">What is the capital of China?
The capital of China is Beijing.
The Eiffel Tower is located in Paris.
Shanghai is a major financial center in China.</textarea>
          </label>

          <div class="projector-row">
            <label class="projector-field" for="projectorProjectionMethod">
              <span class="projector-label">Projection</span>
              <select id="projectorProjectionMethod">
                <option value="umap">UMAP</option>
                <option value="tsne">t-SNE</option>
                <option value="pca">PCA</option>
              </select>
            </label>
            <label class="projector-field" for="projectorMetric">
              <span class="projector-label">Metric</span>
              <select id="projectorMetric">
                <option value="cosine">cosine</option>
                <option value="euclidean">euclidean</option>
              </select>
            </label>
          </div>

          <div class="projector-row">
            <label class="projector-field" for="projectorNeighborsK">
              <span class="projector-label">Neighbors K</span>
              <input id="projectorNeighborsK" type="number" min="1" max="256" value="10" />
            </label>
            <label class="projector-field" for="projectorPointSize">
              <span class="projector-label">Point Size</span>
              <input id="projectorPointSize" type="number" min="1" max="64" step="0.5" value="5" />
            </label>
          </div>

          <div class="projector-row">
            <label class="projector-field" for="projectorInputType">
              <span class="projector-label">Input Type</span>
              <select id="projectorInputType">
                <option value="">document / raw</option>
                <option value="query">query</option>
                <option value="document">document</option>
              </select>
            </label>
            <label class="projector-field" for="projectorModel">
              <span class="projector-label">Model（可选）</span>
              <input id="projectorModel" placeholder="Qwen/Qwen3-Embedding-4B" />
            </label>
          </div>

          <label class="projector-field" for="projectorInstruction">
            <span class="projector-label">Instruction（可选）</span>
            <input id="projectorInstruction" placeholder="Given a web search query, retrieve relevant passages that answer the query" />
          </label>

          <label class="projector-field" for="projectorLabels">
            <span class="projector-label">Labels（可选，每行一条）</span>
            <textarea id="projectorLabels" class="projector-labels" placeholder="news&#10;news&#10;travel&#10;finance"></textarea>
          </label>

          <div class="projector-actions">
            <button class="projector-btn primary" id="projectorRunBtn"><svg class="projector-icon" aria-hidden="true"><use href="#i-run"></use></svg>Run Projector</button>
            <button class="projector-btn secondary" id="projectorDemoBtn"><svg class="projector-icon" aria-hidden="true"><use href="#i-demo"></use></svg>Use Demo</button>
          </div>

          <div class="projector-hint">左键拖拽旋转，滚轮缩放，右键拖拽平移；点选或悬停可联动查看最近邻。</div>
        </div>
      </section>

      <section class="projector-card projector-visualization">
        <div class="projector-card-header projector-visual-header">
          <div>
            <div class="projector-card-title">3D Projection</div>
            <div class="projector-card-subtitle">scatter + nearest-neighbor explorer</div>
          </div>
          <div class="projector-badges">
            <span class="projector-badge" id="projectorBadgeModel">model: -</span>
            <span class="projector-badge warn" id="projectorBadgeState">status: idle</span>
            <span class="projector-badge" id="projectorBadgeCount">points: 0</span>
            <span class="projector-badge" id="projectorBadgeLatency">latency: -</span>
          </div>
        </div>
        <div class="projector-canvas-wrap">
          <div id="projectorPlot"></div>
        </div>
      </section>
    </div>

    <div class="projector-analysis-grid">
      <section class="projector-card">
        <div class="projector-card-header">
          <div class="projector-card-title">Selection & Nearest Neighbors</div>
          <span class="projector-card-subtitle">interactive</span>
        </div>
        <div class="projector-card-body projector-selection-body">
          <div id="projectorStatus" class="projector-status warn">Ready to run projector request.</div>
          <div class="projector-info-box">
            <div class="projector-key">Selected Point</div>
            <div class="projector-value" id="projectorSelectedPoint">None</div>
          </div>
          <div class="projector-info-box projector-neighbors-box">
            <div class="projector-key">Nearest Neighbors</div>
            <div id="projectorNeighbors" class="projector-empty">None</div>
          </div>
        </div>
      </section>

      <section class="projector-card">
        <div class="projector-card-header">
          <div class="projector-card-title">Projection Data</div>
          <span class="projector-card-subtitle">JSON</span>
        </div>
        <div class="projector-card-body projector-data-grid">
          <div>
            <div class="projector-key">Projection Meta</div>
            <pre id="projectorMetaOut">{}</pre>
          </div>
          <div>
            <div class="projector-key">Raw Response</div>
            <pre id="projectorRawOut">{}</pre>
          </div>
        </div>
      </section>
    </div>
  </div>
`;

const els = {
  inputs: app.querySelector("#projectorInputs"),
  projectionMethod: app.querySelector("#projectorProjectionMethod"),
  metric: app.querySelector("#projectorMetric"),
  neighborsK: app.querySelector("#projectorNeighborsK"),
  pointSize: app.querySelector("#projectorPointSize"),
  inputType: app.querySelector("#projectorInputType"),
  model: app.querySelector("#projectorModel"),
  instruction: app.querySelector("#projectorInstruction"),
  labels: app.querySelector("#projectorLabels"),
  runBtn: app.querySelector("#projectorRunBtn"),
  demoBtn: app.querySelector("#projectorDemoBtn"),
  plot: app.querySelector("#projectorPlot"),
  status: app.querySelector("#projectorStatus"),
  selectedPoint: app.querySelector("#projectorSelectedPoint"),
  neighbors: app.querySelector("#projectorNeighbors"),
  metaOut: app.querySelector("#projectorMetaOut"),
  rawOut: app.querySelector("#projectorRawOut"),
  badgeModel: app.querySelector("#projectorBadgeModel"),
  badgeState: app.querySelector("#projectorBadgeState"),
  badgeCount: app.querySelector("#projectorBadgeCount"),
  badgeLatency: app.querySelector("#projectorBadgeLatency"),
};

const state = {
  points: [],
  plotPoints: [],
  neighbors: {},
  labelsByIndex: [],
  categoriesByIndex: [],
  running: false,
  currentResponse: null,
  pointSize: 5,
  selectedIndex: null,
  plotEventsBound: false,
};

const palette = [
  "#60a5fa",
  "#34d399",
  "#f59e0b",
  "#f472b6",
  "#a78bfa",
  "#22d3ee",
  "#e879f9",
  "#f87171",
  "#84cc16",
  "#c084fc",
];

function setStatus(level, text) {
  els.status.classList.remove("ok", "warn", "bad");
  els.status.classList.add(level);
  els.status.textContent = text;

  els.badgeState.classList.remove("ok", "warn", "bad");
  els.badgeState.classList.add(level);
  els.badgeState.textContent = `status: ${text}`;
}

function splitLines(text) {
  return text
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildPayload() {
  const inputs = splitLines(els.inputs.value);
  if (!inputs.length) return null;

  const payload = {
    inputs,
    projection_method: els.projectionMethod.value,
    metric: els.metric.value,
    neighbors_k: Number(els.neighborsK.value || 10),
    point_size: Number(els.pointSize.value || 5),
  };

  if (els.inputType.value) payload.input_type = els.inputType.value;
  if (els.model.value.trim()) payload.model = els.model.value.trim();
  if (els.instruction.value.trim()) payload.instruction = els.instruction.value.trim();

  const labels = splitLines(els.labels.value);
  if (labels.length) {
    if (labels.length === inputs.length) {
      payload.labels = labels;
    } else {
      setStatus("warn", "labels count != inputs count, labels ignored");
    }
  }
  return payload;
}

function makeColorIndex(labels) {
  const categoryMap = new Map();
  let cursor = 0;
  return labels.map((label) => {
    const key = label || "unlabeled";
    if (!categoryMap.has(key)) {
      categoryMap.set(key, cursor++);
    }
    return categoryMap.get(key);
  });
}

function clampToPlotRange(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-1, Math.min(1, value));
}

function normalizePlotPoints(points, labels) {
  const categoryIndex = makeColorIndex(labels);
  return points.map((point, index) => ({
    index,
    text: point.text,
    label: point.label,
    category: categoryIndex[index] || 0,
    x: clampToPlotRange(Number.isFinite(point.normalized_x) ? point.normalized_x : point.x),
    y: clampToPlotRange(Number.isFinite(point.normalized_y) ? point.normalized_y : point.y),
    z: clampToPlotRange(
      Number.isFinite(point.normalized_z)
        ? point.normalized_z
        : Number.isFinite(point.z)
          ? point.z
          : 0
    ),
  }));
}

function renderSelected(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.points.length) {
    els.selectedPoint.textContent = "None";
    els.neighbors.innerHTML = "<div class='projector-empty'>None</div>";
    return;
  }

  const point = state.points[index];
  els.selectedPoint.textContent = `[${point.index}] ${point.text}`;
  const nn = state.neighbors[String(index)] || [];
  if (!nn.length) {
    els.neighbors.innerHTML = "<div class='projector-empty'>No neighbors</div>";
    return;
  }

  const list = nn
    .map((item) => {
      const peer = state.points[item.index];
      const preview = peer ? peer.text : `index ${item.index}`;
      return `<li><strong>#${item.index}</strong> sim=${item.similarity.toFixed(4)}<br/>${preview}</li>`;
    })
    .join("");
  els.neighbors.innerHTML = `<ul>${list}</ul>`;
}

function getRelatedSet(index) {
  if (!Number.isInteger(index) || index < 0) return new Set();
  const related = [index, ...(state.neighbors[String(index)] || []).map((item) => item.index)];
  return new Set(related.filter((item) => Number.isInteger(item) && item >= 0 && item < state.plotPoints.length));
}

function buildMarkerStyle(selectedIndex) {
  const relatedSet = getRelatedSet(selectedIndex);
  const sizes = [];
  const colors = [];

  for (let i = 0; i < state.plotPoints.length; i += 1) {
    const category = state.categoriesByIndex[i] || 0;
    const baseColor = palette[category % palette.length];

    let color = baseColor;
    let size = state.pointSize;

    if (Number.isInteger(selectedIndex) && i === selectedIndex) {
      color = "#f8fafc";
      size = Math.max(state.pointSize + 3, state.pointSize * 1.8);
    } else if (relatedSet.has(i)) {
      color = "#fde68a";
      size = Math.max(state.pointSize + 2, state.pointSize * 1.4);
    }

    sizes.push(size);
    colors.push(color);
  }

  return { sizes, colors };
}

function ensurePlotEvents() {
  if (state.plotEventsBound) return;

  els.plot.on("plotly_click", (event) => {
    const point = event?.points?.[0];
    const index = point?.customdata;
    if (Number.isInteger(index)) {
      highlightPoint(index);
    }
  });

  els.plot.on("plotly_hover", (event) => {
    const point = event?.points?.[0];
    const index = point?.customdata;
    if (Number.isInteger(index)) {
      renderSelected(index);
    }
  });

  state.plotEventsBound = true;
}

function getAxisLength() {
  let maxAbs = 0;
  for (const point of state.plotPoints) {
    maxAbs = Math.max(maxAbs, Math.abs(point.x), Math.abs(point.y), Math.abs(point.z));
  }
  return Math.max(0.45, Math.min(0.95, maxAbs + 0.08));
}

function buildCoordinateAxisTraces() {
  const axisLength = getAxisLength();
  return [
    {
      type: "scatter3d",
      mode: "lines+text",
      x: [0, axisLength],
      y: [0, 0],
      z: [0, 0],
      text: ["", "X"],
      textposition: "top center",
      textfont: { size: 10, color: "#f87171" },
      hoverinfo: "skip",
      line: { color: "#ef4444", width: 5 },
    },
    {
      type: "scatter3d",
      mode: "lines+text",
      x: [0, 0],
      y: [0, axisLength],
      z: [0, 0],
      text: ["", "Y"],
      textposition: "top center",
      textfont: { size: 10, color: "#86efac" },
      hoverinfo: "skip",
      line: { color: "#22c55e", width: 5 },
    },
    {
      type: "scatter3d",
      mode: "lines+text",
      x: [0, 0],
      y: [0, 0],
      z: [0, axisLength],
      text: ["", "Z"],
      textposition: "top center",
      textfont: { size: 10, color: "#93c5fd" },
      hoverinfo: "skip",
      line: { color: "#3b82f6", width: 5 },
    },
  ];
}

function buildOriginRayLineTrace() {
  const x = [];
  const y = [];
  const z = [];
  for (const point of state.plotPoints) {
    x.push(0, point.x, null);
    y.push(0, point.y, null);
    z.push(0, point.z, null);
  }
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z,
    hoverinfo: "skip",
    line: {
      color: "rgba(56,189,248,0.45)",
      width: 3,
    },
  };
}

function buildOriginRayArrowTrace() {
  const x = [];
  const y = [];
  const z = [];

  for (const point of state.plotPoints) {
    const length = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (length < 1e-6) continue;
    const dx = point.x / length;
    const dy = point.y / length;
    const dz = point.z / length;

    const arrowLength = Math.min(0.02, Math.max(0.008, length * 0.028));
    const wingLength = arrowLength * 0.48;
    const up = Math.abs(dz) < 0.92 ? [0, 0, 1] : [0, 1, 0];

    const exRaw = dy * up[2] - dz * up[1];
    const eyRaw = dz * up[0] - dx * up[2];
    const ezRaw = dx * up[1] - dy * up[0];
    const eLen = Math.sqrt(exRaw * exRaw + eyRaw * eyRaw + ezRaw * ezRaw);
    if (eLen < 1e-6) continue;

    const ex = exRaw / eLen;
    const ey = eyRaw / eLen;
    const ez = ezRaw / eLen;

    const tipX = point.x;
    const tipY = point.y;
    const tipZ = point.z;
    const baseX = tipX - dx * arrowLength;
    const baseY = tipY - dy * arrowLength;
    const baseZ = tipZ - dz * arrowLength;

    const wing1X = baseX + ex * wingLength;
    const wing1Y = baseY + ey * wingLength;
    const wing1Z = baseZ + ez * wingLength;
    const wing2X = baseX - ex * wingLength;
    const wing2Y = baseY - ey * wingLength;
    const wing2Z = baseZ - ez * wingLength;

    x.push(tipX, wing1X, null, tipX, wing2X, null);
    y.push(tipY, wing1Y, null, tipY, wing2Y, null);
    z.push(tipZ, wing1Z, null, tipZ, wing2Z, null);
  }

  if (!x.length) return null;
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z,
    hoverinfo: "skip",
    line: {
      color: "rgba(90,225,255,0.9)",
      width: 2,
    },
  };
}

function buildOriginTrace() {
  return {
    type: "scatter3d",
    mode: "markers+text",
    x: [0],
    y: [0],
    z: [0],
    text: ["0"],
    textposition: "top center",
    textfont: { size: 9, color: "#67e8f9" },
    hovertemplate: "0 (0, 0, 0)<extra></extra>",
    marker: {
      size: Math.max(2, Math.ceil(state.pointSize * 0.45)),
      color: "#22d3ee",
      line: { width: 1.2, color: "rgba(8,47,73,0.95)" },
      opacity: 1,
    },
  };
}

async function render3DPlot(selectedIndex = null) {
  if (!state.plotPoints.length) {
    Plotly.purge(els.plot);
    state.plotEventsBound = false;
    return;
  }

  const markerStyle = buildMarkerStyle(selectedIndex);
  const x = state.plotPoints.map((item) => item.x);
  const y = state.plotPoints.map((item) => item.y);
  const z = state.plotPoints.map((item) => item.z);
  const text = state.plotPoints.map((item) => `#${item.index}`);
  const customdata = state.plotPoints.map((item) => item.index);
  const hovertext = state.plotPoints.map((item) => item.text || `#${item.index}`);
  const axisTraces = buildCoordinateAxisTraces();
  const originRayLineTrace = buildOriginRayLineTrace();
  const originRayArrowTrace = buildOriginRayArrowTrace();
  const originTrace = buildOriginTrace();

  const data = [];
  data.push({
    type: "scatter3d",
    mode: "markers+text",
    x,
    y,
    z,
    text,
    textposition: "top center",
    textfont: { size: 11, color: "#cbd5e1" },
    customdata,
    hovertext,
    hovertemplate: "#%{customdata}<br>%{hovertext}<extra></extra>",
    marker: {
      size: markerStyle.sizes,
      color: markerStyle.colors,
      opacity: 0.96,
      line: { width: 1.2, color: "rgba(15,23,42,0.95)" },
    },
  });
  data.push(...axisTraces);
  data.push(originRayLineTrace);
  if (originRayArrowTrace) data.push(originRayArrowTrace);
  data.push(originTrace);

  const layout = {
    autosize: true,
    showlegend: false,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    uirevision: "projector-3d",
    scene: {
      dragmode: "orbit",
      bgcolor: "rgba(2,6,23,0.72)",
      xaxis: { visible: false, showbackground: false },
      yaxis: { visible: false, showbackground: false },
      zaxis: { visible: false, showbackground: false },
      camera: { eye: { x: 1.45, y: 1.2, z: 1.15 } },
      aspectmode: "cube",
    },
  };

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso3d", "select2d", "select3d"],
  };

  await Plotly.react(els.plot, data, layout, config);
  ensurePlotEvents();
}

function updatePlotHighlight(index) {
  if (!state.plotPoints.length) return;
  const markerStyle = buildMarkerStyle(index);
  Plotly.restyle(
    els.plot,
    {
      "marker.size": [markerStyle.sizes],
      "marker.color": [markerStyle.colors],
    },
    [0]
  );
}

function highlightPoint(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.plotPoints.length) {
    return;
  }
  state.selectedIndex = index;
  renderSelected(index);
  updatePlotHighlight(index);
}

async function runProjector() {
  if (state.running) return;
  const payload = buildPayload();
  if (!payload) {
    setStatus("bad", "inputs required");
    return;
  }

  state.running = true;
  els.runBtn.disabled = true;
  setStatus("warn", "requesting");
  const startedAt = performance.now();

  try {
    const response = await fetch("/v1/embeddings/projector", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result?.error?.message || `HTTP ${response.status}`);
    }

    state.currentResponse = result;
    state.points = Array.isArray(result.points) ? result.points : [];
    state.neighbors = result.neighbors || {};
    state.labelsByIndex = state.points.map((point) => point.label || "unlabeled");
    state.plotPoints = normalizePlotPoints(state.points, state.labelsByIndex);
    state.categoriesByIndex = state.plotPoints.map((point) => point.category || 0);
    state.pointSize = Math.max(3, Number(payload.point_size || 5));
    state.selectedIndex = null;

    await render3DPlot(null);

    const latency = Math.round(performance.now() - startedAt);
    els.badgeModel.textContent = `model: ${result.model || "-"}`;
    els.badgeCount.textContent = `points: ${state.points.length}`;
    els.badgeLatency.textContent = `latency: ${latency} ms`;
    els.metaOut.textContent = JSON.stringify(result.projection_meta || {}, null, 2);
    els.rawOut.textContent = JSON.stringify(result, null, 2);
    setStatus("ok", "ready");

    if (state.points.length > 0) {
      highlightPoint(0);
    } else {
      renderSelected(-1);
    }
  } catch (error) {
    setStatus("bad", error instanceof Error ? error.message : String(error));
  } finally {
    state.running = false;
    els.runBtn.disabled = false;
  }
}

function fillDemo() {
  els.inputs.value = [
    "What is the capital of China?",
    "The capital of China is Beijing.",
    "The Eiffel Tower is located in Paris.",
    "Paris is the capital of France.",
    "Shanghai is a major financial center in China.",
    "How to improve retrieval quality for Chinese documents?",
  ].join("\n");
  els.labels.value = ["query", "fact", "fact", "fact", "fact", "query"].join("\n");
  els.projectionMethod.value = "umap";
  els.metric.value = "cosine";
  els.neighborsK.value = "10";
  els.pointSize.value = "5";
  els.inputType.value = "document";
}

els.runBtn.addEventListener("click", runProjector);
els.demoBtn.addEventListener("click", fillDemo);

fillDemo();

  root.dataset.projectorMounted = "true";
  mountedProjector = { root, plot: els.plot };
}

const standaloneRoot = document.getElementById("app");
if (standaloneRoot) {
  mountProjector(standaloneRoot);
}

const embeddedRoot = document.getElementById("projector-root");
if (embeddedRoot) {
  mountProjector(embeddedRoot);
}

window.qwenEmbeddingProjector = { mountProjector, resizeProjector };
