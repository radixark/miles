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
      el("span", { class: "muted" }, [`${lanes.length} trajectories · wheel = zoom · drag = pan · sort`]),
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
  const single = lanes.length === 1; // the L2 page's own lane: no sorting to do
  const panel = el("div", { class: "panel" }, [
    el("h3", {}, [
      single
        ? `sample s${lanes[0].sample_index} lifecycle — generated, waited, tool-called`
        : "batch anatomy — when each sample generated, waited, tool-called",
    ]),
    ...(single ? [] : [sortRow]),
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
  // zoomable time window, same idiom as the timeline lane view: wheel =
  // zoom at cursor, drag = pan, double-click = reset. Purely visual — all
  // segments are already client-side, no refetch.
  let v0 = T0;
  let v1 = T1;
  const X = (t, w) => M_LEFT + ((t - v0) / Math.max(v1 - v0, 1e-9)) * (w - M_LEFT - M_RIGHT);
  const tAt = (clientX, rect) => v0 + ((clientX - rect.left - M_LEFT) / (rect.width - M_LEFT - M_RIGHT)) * (v1 - v0);

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
    const span = v1 - v0;
    const tick = [1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600].find((s) => span / s <= 8) || 7200;
    for (let t = Math.ceil((v0 - T0) / tick) * tick; t <= v1 - T0; t += tick) {
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
        if ((attempt.t1 ?? T1) < v0 || (attempt.t0 ?? T0) > v1) continue;
        ctx.fillStyle = COLORS.attempt;
        const x0 = X(attempt.t0 ?? T0, W);
        ctx.fillRect(x0, y + padAttempt, Math.max(X(attempt.t1 ?? T1, W) - x0, 1), ROW - 2 * padAttempt);
      }
      for (const segment of lane.segments) {
        if ((segment.t1 ?? T1) < v0 || (segment.t0 ?? T0) > v1) continue;
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

    if (consumeTs !== null && consumeTs !== undefined && consumeTs >= v0 && consumeTs <= v1) {
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

  canvas.onwheel = (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const pivot = tAt(ev.clientX, rect);
    const factor = Math.exp(ev.deltaY * 0.002);
    v0 = Math.max(T0, pivot + (v0 - pivot) * factor);
    v1 = Math.min(T1, pivot + (v1 - pivot) * factor);
    if (v1 - v0 < 1) v1 = v0 + 1;
    draw();
  };
  canvas.ondblclick = () => ((v0 = T0), (v1 = T1), draw());
  let dragFrom = null; // {x, v0, v1}; a <4px move on release is a click
  canvas.onmousedown = (ev) => (dragFrom = { x: ev.clientX, v0, v1 });
  canvas.onmouseup = (ev) => {
    const moved = dragFrom && Math.abs(ev.clientX - dragFrom.x) >= 4;
    dragFrom = null;
    if (moved) return;
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((ev.clientY - rect.top - M_TOP) / ROW);
    if (i >= 0 && i < order.length) onClickSample(order[i].sample_index);
  };

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    if (dragFrom) {
      const dt = ((dragFrom.x - ev.clientX) / (rect.width - M_LEFT - M_RIGHT)) * (dragFrom.v1 - dragFrom.v0);
      const span = dragFrom.v1 - dragFrom.v0;
      v0 = Math.max(T0, Math.min(dragFrom.v0 + dt, T1 - span));
      v1 = v0 + span;
      draw();
      return;
    }
    const i = Math.floor((ev.clientY - rect.top - M_TOP) / ROW);
    const x = ev.clientX - rect.left;
    if (i < 0 || i >= order.length || x < M_LEFT || x > rect.width - M_RIGHT) {
      hideTooltip();
      return;
    }
    const lane = order[i];
    const t = tAt(ev.clientX, rect);
    const segment = lane.segments.find((s) => (s.t0 ?? T0) <= t && t < (s.t1 ?? T1));
    const versions = lane.versions.length > 1 ? `  v${lane.versions[0]}–v${lane.versions.at(-1)}` : "";
    const lines = [`s${lane.sample_index}  +${fmtNum(t - T0)}s  ${lane.status}${versions}`];
    if (segment) {
      const version = segment.weight_version ? ` · v${segment.weight_version}` : "";
      const turn = segment.turn > 0 ? `turn ${segment.turn} ` : "";
      lines.push(`${turn}${segment.kind === "gen" ? "generating" : "tool wait"}${version}`);
    } else if (lane.attempts.some((a) => (a.t0 ?? T0) <= t && t < (a.t1 ?? T1))) {
      lines.push("queued / waiting");
    }
    showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
  };
  canvas.onmouseleave = () => {
    hideTooltip();
    dragFrom = null; // leaving the canvas cancels a pan
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
