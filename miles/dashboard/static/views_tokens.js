import { createAnatomy } from "./anatomy.js";
import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { divergingColor, drawChart, hideTooltip, sequentialColor, showTooltip } from "./charts.js";

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
  view.replaceChildren(
    el("p", { class: "muted" }, [
      "loading sample… the first open of a step detokenizes its whole dump and can take several minutes",
    ]),
  );
  const probe = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/tokens`, {
    start: 0,
    end: 1,
    eval: evaluation,
  });
  const total = probe.total_len;

  let windowSize = total <= 4096 ? total : 1024;
  let start = 0;
  let stat = "imp_ratio";

  const controls = el("div", { class: "controls" });
  const strip = el("div", { class: "panel" });
  const chartPanel = el("div", { class: "panel" });

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

    // ---- per-token chart of the selected stat ----
    const canvas = el("canvas", { class: "chart" });
    chartPanel.replaceChildren(el("h3", {}, [`${stat} per token position`]), canvas);
    queueMicrotask(() =>
      drawChart(
        canvas,
        values.map((v, i) => ({
          x: payload.prompt_len + Math.max(0, payload.start - payload.prompt_len) + i,
          y: v,
          label: `pos ${payload.prompt_len + Math.max(0, payload.start - payload.prompt_len) + i}\n${stat} = ${fmtNum(v)}`,
        })).filter((p) => p.y !== null),
      ),
    );
  }

  // this sample's own lifecycle lane (train steps with the trajectory
  // probes; empty for eval samples and probe-less runs -> no panel)
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
  view.replaceChildren(...panels, controls, strip, chartPanel);
  await load();
}
