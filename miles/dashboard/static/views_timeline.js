import { api } from "./api.js";
import { el, fmtNum, setViewCleanup } from "./app.js";
import { createCarpet } from "./carpet.js";
import { hideTooltip, showTooltip } from "./charts.js";

const PHASE_COLORS = {
  initialize: "#b8c4d8",
  rollout: "#1d9e75",
  eval_rollout: "#5dcaa5",
  actor_train: "#7f77dd",
  train_log_probs: "#afa9ec",
  log_probs: "#afa9ec",
  ref_log_probs: "#9f97e0",
  data_preprocess: "#8b95a5",
  train_wait: "#e4e7ec",
  update_weights: "#d85a30",
  ref_model_update: "#b98156",
  save_model: "#8a6d3b",
  sleep: "#d6dbe3",
  wake_up: "#ba7517",
};
const DEFAULT_PHASE_COLOR = "#98a1b0";
const OVERLAY_METRICS = [
  "sglang_num_running_reqs",
  "sglang_gen_throughput",
  "sglang_token_usage",
  "sglang_cache_hit_rate",
];
const OVERLAY_COLOR = "#d97706";
const MEM_COLOR = "#1a9e6e";
const UTIL_STROKE = "rgba(47, 111, 235, 0.9)";
const UTIL_FILL = "rgba(47, 111, 235, 0.14)";
const TEXT = "#24292f";
const MUTED = "#667080";
const GRID = "#e3e7ee";

const FOLLOW_REDRAW_MS = 5000;
const LANE_CAP = 32;
const QUICK_PICKS = [
  { label: "pick: lowest util", criterion: "lowest_util" },
  { label: "pick: slowest update_weights", criterion: "slowest_phase:update_weights" },
];
const QUICK_PICK_K = 8;

const LANE_H = 66;
const PHASE_H = 14;
const UTIL_H = 40;
const M_LEFT = 96;
const M_TOP = 24;
const M_RIGHT = 14;

export async function renderTimeline(view, meta, route) {
  // mutable data state, reloaded on every follow-mode refresh
  let selection = route?.lanes || null; // lane-selection grammar string
  let resolvedKeys = null; // full "node:gpu" set the grammar resolved to (null = no pool)
  let capped = 0; // lanes hidden by LANE_CAP
  let lanes = [];
  let windows = [];
  let phasesByLane = new Map();
  let bubbles = [];
  let gpu = {};
  let engineSeries = [];
  let T0 = 0;
  let T1 = 1;
  let v0 = 0;
  let v1 = 1;
  let haveData = false;
  let overlayMetric = meta.capabilities.has_engine_series ? OVERLAY_METRICS[0] : null;
  let showMem = false;
  let multiNode = false;
  const laneKey = (l) => `${l.node}:${l.gpu}`;

  async function loadData() {
    const [topology, phasesRes, bubblesRes, gpuRes] = await Promise.all([
      api("/api/timeline/topology"),
      api("/api/timeline/phases", { lanes: selection }),
      api("/api/timeline/bubbles"),
      api("/api/timeline/gpu", { max_points: 4000, lanes: selection }),
    ]);
    engineSeries = overlayMetric
      ? (await api("/api/timeline/engine_series", { metric: overlayMetric, max_points: 4000 })).series
      : [];
    lanes = topology.lanes;
    resolvedKeys = null;
    if (selection) {
      // the filtered responses reveal which lanes the grammar resolved to
      const keys = new Set(Object.keys(gpuRes.lanes));
      for (const p of phasesRes.phases) keys.add(`${p.node}:${p.gpu}`);
      lanes = lanes.filter((l) => keys.has(laneKey(l)));
      resolvedKeys = new Set(lanes.map(laneKey));
    }
    capped = Math.max(0, lanes.length - LANE_CAP);
    if (capped) lanes = lanes.slice(0, LANE_CAP);
    windows = topology.windows;
    bubbles = bubblesRes.bubbles;
    gpu = gpuRes.lanes;
    multiNode = new Set(lanes.map((l) => l.node)).size > 1;
    phasesByLane = new Map(lanes.map((l) => [laneKey(l), []]));
    for (const p of phasesRes.phases) phasesByLane.get(`${p.node}:${p.gpu}`)?.push(p);

    const stamps = [];
    for (const p of phasesRes.phases) stamps.push(p.t0, p.t1);
    for (const s of Object.values(gpu)) if (s.ts.length) stamps.push(s.ts[0], s.ts[s.ts.length - 1]);
    if (!stamps.length) {
      haveData = false;
      return;
    }
    // keep following the growing run while the user is on the full range; a
    // zoomed-in window is theirs and stays put
    const wasFullRange = !haveData || (v0 <= T0 + 1e-6 && v1 >= T1 - 1e-6);
    T0 = Math.min(...stamps);
    T1 = Math.max(...stamps);
    if (wasFullRange) {
      v0 = T0;
      v1 = T1;
    }
    haveData = true;
  }

  const engineAt = (node, gpuId, t) => {
    for (const w of windows) {
      if (t >= w.t0 && (w.t1 === null || t < w.t1)) {
        for (const e of w.engines) if (e.gpus.some(([n, g]) => n === node && g === gpuId)) return e.addr;
      }
    }
    return null;
  };

  // ------------------------------- toolbar ----------------------------------
  const toolbar = el("div", { class: "controls" });
  const overlayScale = el("span", { style: `color: ${OVERLAY_COLOR}; font-size: 12px` });
  const followBadge = el("span", { class: "muted" });
  const renderToolbar = () => {
    const chips = OVERLAY_METRICS.map((m) =>
      el(
        "button",
        {
          class: m === overlayMetric ? "active" : "",
          onclick: async () => {
            overlayMetric = m === overlayMetric ? null : m;
            renderToolbar();
            await loadData();
            renderAll();
          },
        },
        [m.replace("sglang_", "")],
      ),
    );
    const memBtn = el(
      "button",
      { class: showMem ? "active" : "", onclick: () => ((showMem = !showMem), renderToolbar(), draw()) },
      ["mem"],
    );
    toolbar.replaceChildren(
      el("span", { class: "muted" }, ["overlay"]),
      ...(meta.capabilities.has_engine_series ? chips : [el("span", { class: "muted" }, ["(no engine series)"])]),
      memBtn,
      el("button", { onclick: () => ((v0 = T0), (v1 = T1), draw(), syncCarpet()) }, ["reset zoom"]),
      overlayScale,
      followBadge,
      el("span", { class: "muted" }, ["wheel = zoom · drag = pan"]),
    );
  };

  // ----------------------------- lane selection -----------------------------
  const selRow = el("div", { class: "controls" });
  const selError = el("span", { class: "error", style: "padding: 0" });
  const selTerms = () => (selection ? selection.split(",").map((s) => s.trim()).filter(Boolean) : []);

  async function setSelection(grammar) {
    selection = grammar || null;
    history.replaceState(null, "", selection ? `#/timeline?lanes=${encodeURIComponent(selection)}` : "#/timeline");
    selError.textContent = "";
    renderSelection();
    try {
      await loadData();
    } catch (err) {
      selError.textContent = String(err);
      return;
    }
    renderAll();
    carpet.redraw(); // selection markers
  }

  function renderSelection() {
    const input = el("input", {
      type: "text",
      placeholder: "add: rank:0-7 · node:<ip> · gpu:<node>:<i> · engine:<addr> · role:train",
      style: "flex: 1; min-width: 280px",
    });
    input.onkeydown = (ev) => {
      if (ev.key === "Enter" && input.value.trim()) {
        setSelection([...selTerms(), input.value.trim()].join(", "));
      }
    };
    selRow.replaceChildren(
      el("span", { class: "muted" }, ["lanes"]),
      ...selTerms().map((term) =>
        el("span", { class: "chip" }, [
          term,
          el(
            "button",
            { onclick: () => setSelection(selTerms().filter((t) => t !== term).join(", ")) },
            ["×"],
          ),
        ]),
      ),
      ...(selection ? [] : [el("span", { class: "muted" }, ["all"])]),
      input,
      ...QUICK_PICKS.map(({ label, criterion }) =>
        el(
          "button",
          {
            onclick: async () => {
              const res = await api("/api/timeline/outliers", { criterion, top_k: QUICK_PICK_K });
              if (res.outliers.length) setSelection(res.outliers.map((o) => `gpu:${o.node}:${o.gpu}`).join(", "));
            },
          },
          [label],
        ),
      ),
      ...(selection ? [el("button", { onclick: () => setSelection(null) }, ["clear"])] : []),
      ...(capped ? [el("span", { class: "warn" }, [`selection has ${capped + LANE_CAP} lanes; showing first ${LANE_CAP}`])] : []),
      selError,
    );
  }

  // -------------------------------- carpet ----------------------------------
  const carpet = createCarpet({
    phaseColors: PHASE_COLORS,
    runStart: () => T0,
    selectedKeys: () => resolvedKeys ?? new Set(),
    // toggle semantics: brushed lanes join the detailed pool; a brush that only
    // covers already-pooled lanes removes them instead
    onBrush: (keys) => {
      const pool = resolvedKeys ?? new Set();
      if (keys.every((k) => pool.has(k))) {
        const kept = [...pool].filter((k) => !keys.includes(k));
        setSelection(kept.map((k) => `gpu:${k}`).join(", "));
      } else {
        const added = keys.filter((k) => !pool.has(k)).map((k) => `gpu:${k}`);
        setSelection([...selTerms(), ...added].join(", "));
      }
    },
  });
  // the carpet tracks the lane view's zoom window (full range -> server default,
  // so a growing follow-mode run keeps extending it)
  const carpetRange = () => (v0 <= T0 + 1e-6 && v1 >= T1 - 1e-6 ? {} : { t0: v0, t1: v1 });
  let carpetTimer = null;
  const syncCarpet = () => {
    clearTimeout(carpetTimer);
    carpetTimer = setTimeout(() => carpet.refresh(carpetRange()).catch(() => {}), 250);
  };

  // ----------------------------- bubble strip -------------------------------
  const bubbleStrip = el("div", { class: "bubblestrip" });
  const renderBubbles = () => {
    if (!bubbles.length) {
      bubbleStrip.replaceChildren();
      return;
    }
    const worst = Math.max(...bubbles.map((b) => b.wait_ratio ?? 0), 0.01);
    bubbleStrip.replaceChildren(el("span", { class: "muted" }, ["wait ratio / step: "]));
    for (const b of bubbles) {
      const cell = el("span", { class: "bubble" }, [String(b.step)]);
      const ratio = b.wait_ratio ?? 0;
      cell.style.background = `rgba(224, 96, 96, ${(0.85 * ratio) / worst})`;
      cell.onclick = () => {
        if (b.step_time !== null) {
          v0 = b.ts - b.step_time;
          v1 = b.ts;
          draw();
          syncCarpet();
        }
      };
      cell.onmousemove = (ev) =>
        showTooltip(
          ev.clientX,
          ev.clientY,
          `step ${b.step}\nwait_ratio = ${fmtNum(ratio)}\nstep_time = ${fmtNum(b.step_time)}s`,
        );
      cell.onmouseleave = hideTooltip;
      bubbleStrip.append(cell);
    }
  };

  // -------------------------------- canvas ----------------------------------
  const canvas = el("canvas", { class: "timeline" });
  const overlayMax = () => Math.max(...engineSeries.flatMap((s) => s.value), 1e-9);
  const memMax = () => Math.max(...Object.values(gpu).flatMap((s) => s.mem_mb), 1);

  function draw() {
    canvas.style.height = `${M_TOP + Math.max(lanes.length, 1) * LANE_H + 8}px`;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const width = rect.width;
    const plotW = width - M_LEFT - M_RIGHT;
    ctx.font = "11px ui-monospace, monospace";
    if (!haveData) {
      ctx.fillStyle = MUTED;
      ctx.fillText("waiting for timeline data…", M_LEFT, M_TOP + 20);
      overlayScale.textContent = "";
      return;
    }
    const X = (t) => M_LEFT + ((t - v0) / (v1 - v0)) * plotW;

    ctx.fillStyle = MUTED;
    ctx.strokeStyle = GRID;
    const span = v1 - v0;
    const tickStep = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600].find((s) => span / s <= 10) || 7200;
    for (let t = Math.ceil((v0 - T0) / tickStep) * tickStep + T0; t <= v1; t += tickStep) {
      const rel = Math.round(t - T0);
      const label = `+${Math.floor(rel / 60)}:${String(rel % 60).padStart(2, "0")}`;
      ctx.fillText(label, X(t) - 12, 12);
      ctx.beginPath();
      ctx.moveTo(X(t), M_TOP - 6);
      ctx.lineTo(X(t), M_TOP + lanes.length * LANE_H);
      ctx.stroke();
    }

    lanes.forEach((lane, i) => {
      const key = laneKey(lane);
      const yPhase = M_TOP + i * LANE_H + 4;
      const yUtilTop = yPhase + PHASE_H + 4;
      const yUtilBot = yUtilTop + UTIL_H;

      ctx.fillStyle = TEXT;
      ctx.fillText(`gpu ${lane.gpu}`, 8, yUtilTop + 4);
      if (multiNode) {
        ctx.fillStyle = MUTED;
        ctx.fillText(lane.node, 8, yUtilTop + 18);
      }
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(M_LEFT, yUtilBot);
      ctx.lineTo(width - M_RIGHT, yUtilBot);
      ctx.stroke();

      for (const p of phasesByLane.get(key) ?? []) {
        if (p.t1 < v0 || p.t0 > v1) continue;
        const x0 = Math.max(X(p.t0), M_LEFT);
        const x1 = Math.min(X(p.t1), width - M_RIGHT);
        ctx.fillStyle = PHASE_COLORS[p.name] ?? DEFAULT_PHASE_COLOR;
        ctx.fillRect(x0, yPhase, Math.max(x1 - x0, 1), PHASE_H);
        if (p.name === "train_wait") {
          // hatch idle explicitly: bubbles must be visible, not blank
          ctx.strokeStyle = MUTED;
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0, yPhase, x1 - x0, PHASE_H);
          ctx.clip();
          for (let x = x0 - PHASE_H; x < x1; x += 5) {
            ctx.beginPath();
            ctx.moveTo(x, yPhase + PHASE_H);
            ctx.lineTo(x + PHASE_H, yPhase);
            ctx.stroke();
          }
          ctx.restore();
        }
      }

      const series = gpu[key];
      if (series && series.ts.length) {
        const yOf = (u) => yUtilBot - (u / 100) * UTIL_H;
        ctx.beginPath();
        let started = false;
        for (let j = 0; j < series.ts.length; j++) {
          const t = series.ts[j];
          if (t < v0 || t > v1) continue;
          const x = X(t);
          const y = yOf(series.util[j]);
          started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          started = true;
        }
        ctx.strokeStyle = UTIL_STROKE;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.lineTo(X(Math.min(series.ts[series.ts.length - 1], v1)), yUtilBot);
        ctx.lineTo(X(Math.max(series.ts[0], v0)), yUtilBot);
        ctx.closePath();
        ctx.fillStyle = UTIL_FILL;
        ctx.fill();

        if (showMem) {
          const scale = memMax();
          ctx.beginPath();
          started = false;
          for (let j = 0; j < series.ts.length; j++) {
            const t = series.ts[j];
            if (t < v0 || t > v1) continue;
            const y = yUtilBot - (series.mem_mb[j] / scale) * UTIL_H;
            started ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y);
            started = true;
          }
          ctx.strokeStyle = MEM_COLOR;
          ctx.stroke();
        }
      }

      if (overlayMetric && engineSeries.length) {
        const scale = overlayMax();
        for (const w of windows) {
          const w0 = Math.max(w.t0, v0);
          const w1 = Math.min(w.t1 ?? v1, v1);
          if (w0 >= w1) continue;
          const addr = engineAt(lane.node, lane.gpu, (w0 + w1) / 2);
          const s = engineSeries.find((es) => es.addr === addr);
          if (!s) continue;
          ctx.beginPath();
          let on = false;
          for (let j = 0; j < s.ts.length; j++) {
            const t = s.ts[j];
            if (t < w0 || t > w1) continue;
            const y = yUtilBot - (s.value[j] / scale) * UTIL_H;
            on ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y);
            on = true;
          }
          ctx.strokeStyle = OVERLAY_COLOR;
          ctx.lineWidth = 1.6;
          ctx.stroke();
          ctx.lineWidth = 1;
        }
      }
    });

    overlayScale.textContent =
      overlayMetric && engineSeries.length
        ? `${overlayMetric.replace("sglang_", "")} scale 0–${fmtNum(overlayMax())}`
        : "";
  }

  function renderAll() {
    renderSelection();
    renderBubbles();
    renderLegend();
    draw();
  }

  // ------------------------------ interactions ------------------------------
  const timeAt = (clientX) => {
    const rect = canvas.getBoundingClientRect();
    return v0 + ((clientX - rect.left - M_LEFT) / (rect.width - M_LEFT - M_RIGHT)) * (v1 - v0);
  };
  canvas.onwheel = (ev) => {
    if (!haveData) return;
    ev.preventDefault();
    const pivot = timeAt(ev.clientX);
    const factor = Math.exp(ev.deltaY * 0.002);
    v0 = Math.max(T0, pivot + (v0 - pivot) * factor);
    v1 = Math.min(T1, pivot + (v1 - pivot) * factor);
    draw();
    syncCarpet();
  };
  let dragFrom = null;
  canvas.onmousedown = (ev) => (dragFrom = { x: ev.clientX, v0, v1 });
  const onMouseUp = () => (dragFrom = null);
  window.addEventListener("mouseup", onMouseUp);
  canvas.ondblclick = () => ((v0 = T0), (v1 = T1), draw(), syncCarpet());
  canvas.onmousemove = (ev) => {
    if (!haveData) return;
    if (dragFrom) {
      const rect = canvas.getBoundingClientRect();
      const dt = ((dragFrom.x - ev.clientX) / (rect.width - M_LEFT - M_RIGHT)) * (dragFrom.v1 - dragFrom.v0);
      const span = dragFrom.v1 - dragFrom.v0;
      v0 = Math.max(T0, Math.min(dragFrom.v0 + dt, T1 - span));
      v1 = v0 + span;
      draw();
      syncCarpet();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const laneIdx = Math.floor((ev.clientY - rect.top - M_TOP) / LANE_H);
    if (laneIdx < 0 || laneIdx >= lanes.length || ev.clientX - rect.left < M_LEFT) {
      hideTooltip();
      return;
    }
    const lane = lanes[laneIdx];
    const key = laneKey(lane);
    const t = timeAt(ev.clientX);
    const lines = [`${key}  +${fmtNum(t - T0)}s`];
    const phase = (phasesByLane.get(key) ?? []).find((p) => p.t0 <= t && t < p.t1);
    if (phase) lines.push(`phase: ${phase.name}${phase.rank >= 0 ? ` (rank ${phase.rank})` : ""}`);
    const series = gpu[key];
    if (series && series.ts.length) {
      let best = 0;
      for (let j = 0; j < series.ts.length; j++) {
        if (Math.abs(series.ts[j] - t) < Math.abs(series.ts[best] - t)) best = j;
      }
      lines.push(`util: ${series.util[best]}%  mem: ${fmtNum(series.mem_mb[best] / 1024)}G`);
    }
    if (overlayMetric) {
      const addr = engineAt(lane.node, lane.gpu, t);
      const s = engineSeries.find((es) => es.addr === addr);
      if (addr && s && s.ts.length) {
        let best = 0;
        for (let j = 0; j < s.ts.length; j++) if (Math.abs(s.ts[j] - t) < Math.abs(s.ts[best] - t)) best = j;
        lines.push(`${overlayMetric.replace("sglang_", "")}: ${fmtNum(s.value[best])}`, `engine: ${addr}`);
      }
    }
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  canvas.onmouseleave = hideTooltip;

  // -------------------------------- legend ----------------------------------
  const legendPanel = el("div", { class: "panel" });
  const renderLegend = () => {
    const present = new Set();
    for (const items of phasesByLane.values()) for (const p of items) present.add(p.name);
    legendPanel.replaceChildren(
      el("h3", {}, ["legend"]),
      el("div", { class: "legend" }, [
        ...[...present].sort().map((name) => {
          const swatch = el("span", {
            class: "bar",
            style: `width: 12px; background: ${PHASE_COLORS[name] ?? DEFAULT_PHASE_COLOR}`,
          });
          return el("span", { style: "display: inline-flex; gap: 4px; align-items: center" }, [swatch, name]);
        }),
        el("span", { style: `color: ${OVERLAY_COLOR}` }, ["— engine overlay"]),
        el("span", { style: `color: ${UTIL_STROKE}` }, ["— gpu util"]),
      ]),
    );
  };

  view.replaceChildren(toolbar, selRow, carpet.root, bubbleStrip, el("div", { class: "panel" }, [canvas]), legendPanel);
  renderToolbar();
  await Promise.all([loadData(), carpet.refresh({})]);
  renderAll();
  carpet.redraw(); // selection markers need the loaded lane list
  const onResize = () => {
    draw();
    carpet.redraw();
  };
  window.addEventListener("resize", onResize);

  // ------------------------- follow-mode auto refresh -----------------------
  let refreshing = false;
  let intervalId = null;
  if (meta.mode === "follow") {
    followBadge.textContent = `live · redraws every ${FOLLOW_REDRAW_MS / 1000}s`;
    intervalId = setInterval(async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        await loadData();
        renderAll();
        await carpet.refresh(carpetRange());
      } catch {
        // transient fetch failure (server restart, network blip): keep polling
      } finally {
        refreshing = false;
      }
    }, FOLLOW_REDRAW_MS);
  }
  setViewCleanup(() => {
    if (intervalId !== null) clearInterval(intervalId);
    clearTimeout(carpetTimer);
    carpet.destroy();
    window.removeEventListener("resize", onResize);
    window.removeEventListener("mouseup", onMouseUp);
  });
}
