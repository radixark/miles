import { createAnatomy } from "./anatomy.js";
import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { divergingColor, drawChart, hideTooltip, sequentialColor, showTooltip } from "./charts.js";
import { renderConversation } from "./conversation.js";

const WINDOW_SIZES = [256, 1024, 4096];

// stat -> how to color a value: diverging stats define center/scale via values
const STATS = {
  imp_ratio: { color: (v) => divergingColor(Math.log(Math.max(v, 1e-6))) },
  lp_diff: { diverging: true },
  advantages: { diverging: true },
  returns: { diverging: true },
  entropy: { sequential: true },
  ref_entropy: { sequential: true },
  train_log_probs: { sequential: true, negate: true },
  rollout_log_probs: { sequential: true, negate: true },
  ref_log_probs: { sequential: true, negate: true },
};

function colorFor(stat, values) {
  const spec = STATS[stat];
  if (spec.color) return spec.color;
  const finite = values.filter((v) => v !== null && Number.isFinite(v));
  if (spec.diverging) {
    const scale = Math.max(...finite.map(Math.abs), 1e-9);
    return (v) => divergingColor(v / scale);
  }
  const transformed = spec.negate ? finite.map((v) => -v) : finite;
  const [lo, hi] = [Math.min(...transformed), Math.max(...transformed)];
  return (v) => sequentialColor(((spec.negate ? -v : v) - lo) / Math.max(hi - lo, 1e-9));
}

export async function renderTokens(view, meta, route) {
  const { rolloutId, sampleIndex, evaluation } = route;
  view.replaceChildren(el("p", { class: "muted" }, ["loading sample…"]));

  // cheap panels first: the lifecycle lane (telemetry) and the conversation
  // (trajectory sidecar); the token machinery costs a full dump load plus
  // detokenize, so it only starts when its tab is opened
  const panels = [];
  if (!evaluation) {
    try {
      const trajectories = await api(`/api/rollout/${rolloutId}/trajectories`, { sample_index: sampleIndex });
      if (trajectories.lanes.length) {
        panels.push(
          createAnatomy({
            lanes: trajectories.lanes,
            consumeTs: trajectories.consume_ts,
            rowsByIndex: new Map(),
            onClickSample: () => {},
          }),
        );
      }
    } catch {
      /* endpoint absent or no events: token view stands alone */
    }
  }

  let conversationRow = null;
  try {
    conversationRow = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/messages`, { eval: evaluation });
  } catch (err) {
    if (!String(err).includes("404")) throw err; // 404 = run recorded no conversation
  }

  const tokensPane = el("div");
  let tokensStarted = false;
  const startTokens = () => {
    if (tokensStarted) return;
    tokensStarted = true;
    tokensPane.replaceChildren(
      el("p", { class: "muted" }, [
        "loading tokens… the first open of a step detokenizes its whole dump and can take several minutes",
      ]),
    );
    loadTokensPane(tokensPane, rolloutId, sampleIndex, evaluation).catch((err) => {
      tokensPane.replaceChildren(el("div", { class: "error" }, [String(err)]));
    });
  };

  if (conversationRow === null) {
    view.replaceChildren(...panels, tokensPane);
    startTokens();
    return;
  }

  const conversationPane = renderConversation(conversationRow);
  const tabs = el("div", { class: "tabs" });
  const body = el("div");
  const select = (name) => {
    tabs.replaceChildren(
      ...["conversation", "tokens"].map((tab) =>
        el("button", { class: tab === name ? "active" : "", onclick: () => select(tab) }, [tab]),
      ),
    );
    body.replaceChildren(name === "conversation" ? conversationPane : tokensPane);
    if (name === "tokens") startTokens();
  };
  view.replaceChildren(...panels, tabs, body);
  select("conversation");
}

async function loadTokensPane(root, rolloutId, sampleIndex, evaluation) {
  const probe = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/tokens`, {
    start: 0,
    end: 1,
    eval: evaluation,
  });
  const total = probe.total_len;

  let windowSize = total <= 4096 ? total : 1024;
  let start = 0;
  let stat = "imp_ratio";
  let chartMetric = "imp_ratio";
  let chartData = null; // full-range payload: the chart spans the whole response

  const controls = el("div", { class: "controls" });
  const strip = el("div", { class: "panel" });
  const chartPanel = el("div", { class: "panel" });
  const chartCanvas = el("canvas", { class: "chart" }); // persistent: zoom survives metric switches

  function renderChart() {
    const available = Object.keys(STATS).filter((s) => chartData[s] !== null && chartData[s] !== undefined);
    if (!available.length) {
      chartPanel.replaceChildren();
      return;
    }
    if (!available.includes(chartMetric)) chartMetric = available[0];
    const chips = available.map((s) =>
      el(
        "button",
        {
          class: s === chartMetric ? "active" : "",
          onclick: () => {
            chartMetric = s;
            if (chartCanvas._zoom) chartCanvas._zoom = { x: chartCanvas._zoom.x, y: null };
            renderChart();
          },
        },
        [s],
      ),
    );
    chartPanel.replaceChildren(
      el("h3", {}, ["per-token metrics"]),
      el("div", { class: "controls" }, [
        ...chips,
        el("span", { class: "muted" }, ["drag = zoom to token range · double-click = reset"]),
      ]),
      chartCanvas,
    );
    const values = chartData[chartMetric];
    const points = [];
    values.forEach((v, i) => {
      if (v === null) return;
      const pos = chartData.prompt_len + i;
      points.push({ x: pos, y: v, label: `pos ${pos}\n${chartMetric} = ${fmtNum(v)}` });
    });
    queueMicrotask(() => drawChart(chartCanvas, points));
  }

  async function loadChart() {
    chartData = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/tokens`, {
      start: 0,
      end: total,
      eval: evaluation,
    });
    renderChart();
  }

  async function load() {
    const payload = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/tokens`, {
      start,
      end: Math.min(start + windowSize, total),
      eval: evaluation,
    });
    const available = Object.keys(STATS).filter((s) => payload[s] !== null && payload[s] !== undefined);
    if (!available.includes(stat)) stat = available[0];

    // ---- controls ----
    const statSelect = el("select", {}, available.map((s) =>
      Object.assign(el("option", { value: s }, [s]), { selected: s === stat }),
    ));
    statSelect.onchange = () => {
      stat = statSelect.value;
      load();
    };
    const sizeSelect = el("select", {}, WINDOW_SIZES.filter((w) => w < total).concat([total]).map((w) =>
      Object.assign(el("option", { value: w }, [w === total ? `all ${total}` : String(w)]), {
        selected: w === windowSize,
      }),
    ));
    sizeSelect.onchange = () => {
      windowSize = Number(sizeSelect.value);
      load();
    };
    const startInput = el("input", { type: "number", value: start, min: 0, max: total - 1 });
    startInput.onchange = () => {
      start = Math.max(0, Math.min(Number(startInput.value), total - 1));
      load();
    };
    controls.replaceChildren(
      el("button", { onclick: () => ((start = Math.max(0, start - windowSize)), load()) }, ["◀"]),
      el("span", {}, [`tokens ${payload.start}–${payload.end} of ${total} (prompt ${payload.prompt_len})`]),
      el("button", { onclick: () => ((start = Math.min(total - 1, start + windowSize)), load()) }, ["▶"]),
      el("span", { class: "muted" }, [" window"]),
      sizeSelect,
      el("span", { class: "muted" }, [" start"]),
      startInput,
      el("span", { class: "muted" }, [" color by"]),
      statSelect,
    );

    // ---- token strip ----
    const values = payload[stat] ?? [];
    const color = values.length ? colorFor(stat, values) : () => "transparent";
    const spans = payload.token_ids.map((tokenId, i) => {
      const respPos = i - payload.response_offset; // index into stat arrays
      const inResponse = respPos >= 0 && respPos < (payload.train_log_probs ?? payload.rollout_log_probs ?? []).length;
      const text = payload.token_text ? payload.token_text[i] : `·${tokenId}`;
      const masked = inResponse && payload.loss_mask !== null && payload.loss_mask[respPos] === 0;
      const value = inResponse && values.length ? values[respPos] : null;
      const span = el("span", { class: `tok ${inResponse ? "" : "prompt"} ${masked ? "masked" : ""}` }, [
        text.replaceAll("\n", "⏎\n"),
      ]);
      if (value !== null && value !== undefined && !masked) span.style.background = color(value);
      span.onmousemove = (ev) => {
        const lines = [`#${payload.start + i} id=${tokenId} ${JSON.stringify(text)}`];
        if (!inResponse) lines.push("(prompt)");
        else {
          for (const s of available) lines.push(`${s} = ${fmtNum(payload[s][respPos])}`);
          if (masked) lines.push("loss_mask = 0");
        }
        showTooltip(ev.clientX, ev.clientY, lines.join("\n"));
      };
      span.onmouseleave = hideTooltip;
      return span;
    });
    strip.replaceChildren(
      el("h3", {}, [`tokens — colored by ${stat}${payload.token_text ? "" : " (no tokenizer: ids shown)"}`]),
      el("div", { class: "tokens" }, spans),
    );

  }

  root.replaceChildren(controls, strip, chartPanel);
  await Promise.all([load(), loadChart()]);
}
