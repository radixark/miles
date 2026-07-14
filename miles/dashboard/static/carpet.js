import { apiBinary } from "./api.js";
import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";

// margins match views_timeline so the carpet and the lane view share an x domain
const M_LEFT = 96;
const M_RIGHT = 14;
const X_BUCKETS = 1200;
const METRICS = ["util", "phase", "mem_mb", "power_w", "lifecycle"];
const LIFECYCLE_COLORS = { queue: "#cfd6e0", generating: "#1d9e75", tool_wait: "#d97706" };
const MAX_LABELED_ROWS = 48;

// sequential ramp for magnitude metrics: one hue, light -> dark
const RAMP_LO = [238, 241, 246];
const RAMP_HI = [22, 64, 143];
const NO_DATA = [245, 246, 248]; // page bg: absence recedes
const MUTED = "#667080";
const TEXT = "#24292f";
const ACCENT = "#2f6feb";

// carpet-only wake_up shift: the lane view's #ba7517 is deutan-indistinguishable
// from update_weights (dE 5.9) without position/width context, so the carpet
// moves it toward yellow (dE 20.7 against update_weights, validator-checked)
const CARPET_WAKE_UP = "#d4a017";

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// createCarpet returns {root, refresh({t0?, t1?}), redraw, destroy}. The carpet
// is the overview over ALL lanes and the row-brush selector; opts:
//   phaseColors     lane-view palette (wake_up overridden here)
//   runStart()      -> run start ts, so tooltips share the lane view's +m:ss origin
//   selectedKeys()  -> Set("node:gpu") for the left-edge selection markers
//   onBrush(keys)   brushed lane keys (click or row drag); caller decides add/remove
export function createCarpet(opts) {
  const phaseColors = { ...opts.phaseColors, wake_up: CARPET_WAKE_UP };
  let metric = METRICS[0];
  let data = null; // last heatmap response
  let range = {};
  let brush = null; // {r0, r1} while dragging

  const canvas = el("canvas", { class: "carpet" });
  const scaleLabel = el("span", { class: "muted", style: "font-size: 12px" });
  const chipsRow = el("div", { class: "controls" });
  const root = el("div", { class: "panel" }, [chipsRow, canvas]);

  const renderChips = () => {
    chipsRow.replaceChildren(
      el("span", { class: "muted" }, ["carpet"]),
      ...METRICS.map((m) =>
        el(
          "button",
          {
            class: m === metric ? "active" : "",
            onclick: () => {
              metric = m;
              renderChips();
              refresh(range);
            },
          },
          [m === "lifecycle" ? "traj" : m],
        ),
      ),
      scaleLabel,
      el("span", { class: "muted" }, ["click gpu = add/remove · drag = brush"]),
    );
  };

  async function refresh(newRange = range) {
    range = newRange;
    data = await apiBinary("/api/timeline/heatmap", {
      metric,
      x_buckets: X_BUCKETS,
      t0: range.t0,
      t1: range.t1,
    });
    redraw();
  }

  const rowH = () => {
    const n = data.rows.length;
    return n <= 16 ? 14 : n <= 64 ? 8 : n <= 256 ? 4 : 2;
  };

  function redraw() {
    if (!data) return;
    const rows = data.rows.length;
    const h = Math.max(rows * (rows ? rowH() : 1), 24);
    canvas.style.height = `${h}px`;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.font = "11px ui-monospace, monospace";
    if (!rows) {
      ctx.fillStyle = MUTED;
      ctx.fillText("waiting for timeline data…", M_LEFT, 16);
      return;
    }
    const plotW = rect.width - M_LEFT - M_RIGHT;

    // 256-entry color lookup: palette ids for phase, ramp for magnitudes
    const lut = new Array(256).fill(NO_DATA);
    if (data.palette) {
      data.palette.forEach((name, i) => {
        if (i > 0) lut[i] = hexToRgb(LIFECYCLE_COLORS[name] ?? phaseColors[name] ?? "#98a1b0");
      });
    } else {
      for (let v = 0; v < 256; v++) {
        lut[v] = RAMP_LO.map((lo, c) => Math.round(lo + ((RAMP_HI[c] - lo) * v) / 255));
      }
      lut[0] = NO_DATA;
    }

    const img = new ImageData(data.x_buckets, rows);
    for (let i = 0; i < data.values.length; i++) {
      const [r, g, b] = lut[data.values[i]];
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    const off = document.createElement("canvas");
    off.width = data.x_buckets;
    off.height = rows;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, data.x_buckets, rows, M_LEFT, 0, plotW, h);

    const sampleRows = data.rows[0]?.sample_index !== undefined;
    if (!sampleRows) {
      // node boundaries (solid) and train/rollout role changes (dashed)
      ctx.strokeStyle = MUTED;
      const roleKey = (row) => row.roles.join(",");
      for (let r = 1; r < rows; r++) {
        const prev = data.rows[r - 1];
        const cur = data.rows[r];
        if (prev.node !== cur.node) ctx.setLineDash([]);
        else if (roleKey(prev) !== roleKey(cur)) ctx.setLineDash([3, 3]);
        else continue;
        ctx.beginPath();
        ctx.moveTo(M_LEFT, r * rowH());
        ctx.lineTo(rect.width - M_RIGHT, r * rowH());
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    const labelEvery = Math.ceil(rows / MAX_LABELED_ROWS);
    ctx.fillStyle = TEXT;
    for (let r = 0; r < rows; r += labelEvery) {
      const label = sampleRows ? `s${data.rows[r].sample_index}` : `g${data.rows[r].index}`;
      ctx.fillText(label, 8, r * rowH() + Math.min(rowH(), 11));
    }

    if (!sampleRows) {
      // left-edge markers for lanes in the current selection
      const selected = opts.selectedKeys();
      ctx.fillStyle = ACCENT;
      for (let r = 0; r < rows; r++) {
        if (selected.has(`${data.rows[r].node}:${data.rows[r].gpu}`)) {
          ctx.fillRect(M_LEFT - 6, r * rowH(), 3, rowH());
        }
      }
    }

    if (brush) {
      const [a, b] = [Math.min(brush.r0, brush.r1), Math.max(brush.r0, brush.r1)];
      ctx.fillStyle = "rgba(47, 111, 235, 0.18)";
      ctx.fillRect(M_LEFT, a * rowH(), plotW, (b - a + 1) * rowH());
    }

    scaleLabel.textContent = data.scale
      ? `0–${fmtNum(data.scale.max)}`
      : data.rows_total > data.rows.length
        ? `showing ${data.rows.length}/${data.rows_total} trajectories`
        : "";
  }

  const rowAt = (clientY) => {
    const rect = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(data.rows.length - 1, Math.floor((clientY - rect.top) / rowH())));
  };

  canvas.onmousedown = (ev) => {
    if (!data || !data.rows.length) return;
    if (data.rows[0].sample_index !== undefined) return; // trajectories: nothing to select
    const r = rowAt(ev.clientY);
    brush = { r0: r, r1: r };
    redraw();
  };
  const onMouseUp = () => {
    if (!brush) return;
    const [a, b] = [Math.min(brush.r0, brush.r1), Math.max(brush.r0, brush.r1)];
    const keys = data.rows.slice(a, b + 1).map((row) => `${row.node}:${row.gpu}`);
    brush = null;
    redraw();
    opts.onBrush(keys);
  };
  window.addEventListener("mouseup", onMouseUp);

  canvas.onmousemove = (ev) => {
    if (!data || !data.rows.length) return;
    if (brush) {
      brush.r1 = rowAt(ev.clientY);
      redraw();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    if (x < M_LEFT || x > rect.width - M_RIGHT) {
      hideTooltip();
      return;
    }
    const r = rowAt(ev.clientY);
    const row = data.rows[r];
    const col = Math.min(
      data.x_buckets - 1,
      Math.floor(((x - M_LEFT) / (rect.width - M_LEFT - M_RIGHT)) * data.x_buckets),
    );
    const value = data.values[r * data.x_buckets + col];
    const t = data.t0 + ((col + 0.5) / data.x_buckets) * (data.t1 - data.t0);
    const identity =
      row.sample_index !== undefined
        ? `s${row.sample_index} (group ${row.group_index})`
        : `g${row.index} ${row.node}:${row.gpu} (${row.roles.join("+") || "?"})`;
    const lines = [`${identity}  +${fmtNum(t - opts.runStart())}s`];
    if (data.palette) lines.push(`phase: ${data.palette[value] || "—"}`);
    else lines.push(`${metric}: ${fmtNum((value / 255) * data.scale.max)}`);
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  canvas.onmouseleave = hideTooltip;

  renderChips();
  return {
    root,
    refresh,
    redraw,
    destroy: () => window.removeEventListener("mouseup", onMouseUp),
  };
}
