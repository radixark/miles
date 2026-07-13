import { el, fmtNum } from "./app.js";
import { hideTooltip, showTooltip } from "./charts.js";

// batch anatomy swimlanes (design §18.4, visual contract: batch_anatomy_demo)
const COLORS = {
  gen: "#1d9e75",
  tool: "#d97706",
  attempt: "#e7eaef", // queue/underlay: idle recedes, activity pops
  consume: "#2f6feb",
  text: "#24292f",
  muted: "#667080",
  stale: "#d85a30",
};
const M_TOP = 18;
// two densities, same philosophy as the rank carpet (§15): full detail only
// while every row can carry text; above that the pane compresses to a carpet
// and the sort chips + tooltip + table below carry identification
const DETAIL_MAX = 48;
const MAX_PANE_PX = 640;

// createAnatomy renders the per-step trajectory swimlanes: one row per sample,
// x = wall clock from the earliest lifecycle event to the consume anchor.
// rowsByIndex supplies the dump-side columns (versions/turns/tools/reward).
export function createAnatomy({ lanes, consumeTs, rowsByIndex, onClickSample }) {
  const detailed = lanes.length <= DETAIL_MAX;
  const ROW = detailed ? 13 : lanes.length <= 512 ? 4 : 2;
  const M_LEFT = detailed ? 150 : 64;
  const M_RIGHT = detailed ? 64 : 14;

  const dumpRow = (lane) => rowsByIndex.get(lane.sample_index) ?? {};
  const SORTS = {
    submit: (l) => l.first_ts,
    staleness: (l) => -(l.versions.length > 1 ? l.versions.at(-1) - l.versions[0] : 0),
    "wall span": (l) => -(l.last_ts - l.first_ts),
    reward: (l) => dumpRow(l).raw_reward ?? dumpRow(l).reward ?? 0,
    turns: (l) => -(dumpRow(l).turns ?? 0),
  };
  let sortKey = "submit";
  let order = [...lanes];
  const resort = () => {
    order = [...lanes].sort((a, b) => SORTS[sortKey](a) - SORTS[sortKey](b) || a.first_ts - b.first_ts);
  };

  const canvas = el("canvas", { class: "anatomy" });
  const wrap = el("div", { class: "anatomywrap" }, [canvas]);
  const sortRow = el("div", { class: "controls" });
  const renderSort = () => {
    sortRow.replaceChildren(
      el("span", { class: "muted" }, [`${lanes.length} trajectories · sort`]),
      ...Object.keys(SORTS).map((key) =>
        el(
          "button",
          {
            class: key === sortKey ? "active" : "",
            onclick: () => {
              sortKey = key;
              renderSort();
              resort();
              draw();
            },
          },
          [key],
        ),
      ),
    );
  };
  const panel = el("div", { class: "panel" }, [
    el("h3", {}, ["batch anatomy — when each sample generated, waited, tool-called"]),
    sortRow,
    wrap,
    el("div", { class: "legend" }, [
      legendSwatch(COLORS.gen, "generating"),
      legendSwatch(COLORS.tool, "tool wait"),
      legendSwatch(COLORS.attempt, "queue/attempt"),
      el("span", { style: `color: ${COLORS.stale}` }, [detailed ? "" : "▏mixed versions"]),
      el("span", { style: `color: ${COLORS.consume}` }, ["│ consume"]),
    ]),
  ]);

  const T0 = Math.min(...lanes.map((l) => l.first_ts));
  const T1 = consumeTs ?? Math.max(...lanes.map((l) => l.last_ts));
  const X = (t, w) => M_LEFT + ((t - T0) / Math.max(T1 - T0, 1e-9)) * (w - M_LEFT - M_RIGHT);

  function draw() {
    const height = M_TOP + order.length * ROW + 6;
    canvas.style.height = `${height}px`;
    wrap.style.maxHeight = `${Math.min(height, MAX_PANE_PX)}px`;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const W = rect.width;
    ctx.font = "10.5px ui-monospace, monospace";

    ctx.fillStyle = COLORS.muted;
    const span = T1 - T0;
    const tick = [10, 30, 60, 120, 300, 600, 1800, 3600].find((s) => span / s <= 8) || 7200;
    for (let t = 0; t <= span; t += tick) {
      ctx.fillText(`+${Math.floor(t / 60)}:${String(Math.round(t) % 60).padStart(2, "0")}`, X(T0 + t, W) - 10, 10);
    }

    const labelEvery = detailed ? 1 : Math.ceil(order.length / 40);
    order.forEach((lane, i) => {
      const y = M_TOP + i * ROW;
      const row = dumpRow(lane);
      const mixed = lane.versions.length > 1;
      if (i % labelEvery === 0) {
        ctx.fillStyle = COLORS.text;
        ctx.fillText(`s${lane.sample_index}`, 8, y + Math.min(ROW, 10));
      }
      if (detailed) {
        const versions = mixed ? `v${lane.versions[0]}–v${lane.versions.at(-1)}` : (row.versions ?? "");
        ctx.fillStyle = mixed ? COLORS.stale : COLORS.muted;
        ctx.fillText(versions, 52, y + 10);
      } else if (mixed) {
        ctx.fillStyle = COLORS.stale; // compact rows: staleness as a left tick
        ctx.fillRect(M_LEFT - 5, y, 3, ROW);
      }

      const padAttempt = detailed ? 4 : ROW > 2 ? 1 : 0;
      const padSegment = detailed ? 2 : 0;
      for (const attempt of lane.attempts) {
        ctx.fillStyle = COLORS.attempt;
        const x0 = X(attempt.t0 ?? T0, W);
        ctx.fillRect(x0, y + padAttempt, Math.max(X(attempt.t1 ?? T1, W) - x0, 1), ROW - 2 * padAttempt);
      }
      for (const segment of lane.segments) {
        ctx.fillStyle = segment.kind === "gen" ? COLORS.gen : COLORS.tool;
        const x0 = X(segment.t0 ?? T0, W);
        ctx.fillRect(x0, y + padSegment, Math.max(X(segment.t1 ?? T1, W) - x0, 1.5), ROW - 2 * padSegment);
      }
      if (detailed) {
        const info = [
          row.turns !== undefined && row.turns !== null ? `${row.turns}t` : null,
          row.tool_calls ? `${row.tool_calls}x` : null,
          row.raw_reward !== undefined && row.raw_reward !== null ? `r=${fmtNum(row.raw_reward)}` : null,
        ]
          .filter(Boolean)
          .join("·");
        ctx.fillStyle = COLORS.muted;
        ctx.fillText(info, W - M_RIGHT + 6, y + 10);
      }
    });

    if (consumeTs !== null && consumeTs !== undefined) {
      const x = X(consumeTs, W);
      ctx.strokeStyle = COLORS.consume;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, M_TOP - 6);
      ctx.lineTo(x, M_TOP + order.length * ROW);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((ev.clientY - rect.top - M_TOP) / ROW);
    const x = ev.clientX - rect.left;
    if (i < 0 || i >= order.length || x < M_LEFT || x > rect.width - M_RIGHT) {
      hideTooltip();
      return;
    }
    const lane = order[i];
    const t = T0 + ((x - M_LEFT) / (rect.width - M_LEFT - M_RIGHT)) * (T1 - T0);
    const segment = lane.segments.find((s) => (s.t0 ?? T0) <= t && t < (s.t1 ?? T1));
    const versions = lane.versions.length > 1 ? `  v${lane.versions[0]}–v${lane.versions.at(-1)}` : "";
    const lines = [`s${lane.sample_index}  +${fmtNum(t - T0)}s  ${lane.status}${versions}`];
    if (segment) {
      const version = segment.weight_version ? ` · v${segment.weight_version}` : "";
      lines.push(`turn ${segment.turn} ${segment.kind === "gen" ? "generating" : "tool wait"}${version}`);
    } else if (lane.attempts.some((a) => (a.t0 ?? T0) <= t && t < (a.t1 ?? T1))) {
      lines.push("queued / waiting");
    }
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  canvas.onmouseleave = hideTooltip;
  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((ev.clientY - rect.top - M_TOP) / ROW);
    if (i >= 0 && i < order.length) onClickSample(order[i].sample_index);
  };

  renderSort();
  resort();
  queueMicrotask(draw);
  window.addEventListener("resize", draw);
  return panel;
}

function legendSwatch(color, label) {
  const swatch = el("span", { class: "bar", style: `width: 12px; background: ${color}` });
  return el("span", { style: "display: inline-flex; gap: 4px; align-items: center" }, [swatch, label]);
}
