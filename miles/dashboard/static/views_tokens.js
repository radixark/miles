import { createAnatomy } from "./anatomy.js";
import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { divergingColor, drawChart, hideTooltip, sequentialColor, showTooltip } from "./charts.js";
import { renderConversation, renderSampleChips } from "./conversation.js";

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

async function groupNavPanel(rolloutId, sampleIndex, evaluation) {
  const { rows } = await api(`/api/rollout/${rolloutId}/summary`, { eval: evaluation });
  const groupIndex = rows.find((r) => r.sample_index === sampleIndex)?.group_index;
  const siblings = groupIndex == null ? [] : rows.filter((r) => r.group_index === groupIndex).map((r) => r.sample_index).sort((a, b) => a - b);
  if (siblings.length < 2) return null; // no group, or a lone sample: nothing to navigate
  const position = siblings.indexOf(sampleIndex);
  const goto = (idx) => (location.hash = `#/rollout/${rolloutId}/sample/${idx}${evaluation ? "?eval=1" : ""}`);
  return el("div", { class: "controls" }, [
    el("button", { onclick: () => position > 0 && goto(siblings[position - 1]) }, ["◀ Prev in group"]),
    el("span", {}, [`Group ${groupIndex} · sample ${position + 1}/${siblings.length}`]),
    el("button", { onclick: () => position < siblings.length - 1 && goto(siblings[position + 1]) }, ["Next in group ▶"]),
  ]);
}

export async function renderTokens(view, meta, route) {
  const { rolloutId, sampleIndex, evaluation } = route;
  view.replaceChildren(el("p", { class: "muted" }, ["loading sample…"]));

  // group nav reuses the (parquet-cached, cheap) summary rows already fetched
  // for the step's samples tab — no detokenize needed to jump between siblings
  const groupNav = await groupNavPanel(rolloutId, sampleIndex, evaluation);

  // cheap panels first: the lifecycle lane (telemetry) and the conversation
  // (trajectory sidecar); the token machinery costs a full dump load plus
  // detokenize, so it only starts when its tab is opened
  const panels = groupNav ? [groupNav] : [];
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
    view.replaceChildren(
      ...panels,
      el("p", { class: "muted" }, [
        "No conversation recorded for this sample — aborted before any model call, or the step's trajectory sidecar is not written yet. Token view only.",
      ]),
      tokensPane,
    );
    startTokens();
    return;
  }

  const conversationPane = renderConversation(conversationRow);
  const tabs = el("div", { class: "tabs" });
  // both panes stay mounted and are toggled with `hidden`, because detaching a
  // pane destroys its layout and the browser resets the token strip's scroll
  // offset to 0 on re-attach — one glance at the conversation would otherwise
  // send the reader back to token 0 of a several-thousand-token prompt
  const body = el("div", {}, [conversationPane, tokensPane]);
  const select = (name) => {
    tabs.replaceChildren(
      ...["conversation", "tokens"].map((tab) =>
        el("button", { class: tab === name ? "active" : "", onclick: () => select(tab) }, [
          tab[0].toUpperCase() + tab.slice(1),
        ]),
      ),
    );
    conversationPane.hidden = name !== "conversation";
    tokensPane.hidden = name !== "tokens";
    if (name === "tokens") {
      startTokens();
      tokensPane._onShown?.(); // work the load deferred because the pane was hidden
    }
  };
  // chips sit above the tab bar, not inside either pane, so status/reward stay
  // on screen while reading tokens
  view.replaceChildren(...panels, renderSampleChips(conversationRow), tabs, body);
  select("conversation");
}

async function loadTokensPane(root, rolloutId, sampleIndex, evaluation) {
  // one full-range read feeds both panes. The chart always spanned the whole
  // response, so the windowed strip fetch this replaces was re-requesting a
  // slice of bytes the server was already sending.
  const payload = await api(`/api/rollout/${rolloutId}/sample/${sampleIndex}/tokens`, { eval: evaluation });
  const available = Object.keys(STATS).filter((s) => payload[s] !== null && payload[s] !== undefined);

  let stat = "imp_ratio";
  if (!available.includes(stat)) stat = available[0];
  let chartMetric = "imp_ratio";

  const controls = el("div", { class: "controls" });
  const strip = el("div", { class: "panel" });
  const chartPanel = el("div", { class: "panel" });
  const chartCanvas = el("canvas", { class: "chart" }); // persistent: zoom survives metric switches

  function renderChart() {
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
      el("h3", {}, ["Per-token metrics"]),
      el("div", { class: "controls" }, [
        ...chips,
        el("span", { class: "muted" }, ["drag = zoom to token range · double-click = reset"]),
      ]),
      chartCanvas,
    );
    const values = payload[chartMetric];
    const points = [];
    let afterGap = false;
    values.forEach((v, i) => {
      if (v === null) {
        afterGap = true;
        return;
      }
      const pos = payload.prompt_len + i;
      points.push({ x: pos, y: v, gap: afterGap, label: `pos ${pos}\n${chartMetric} = ${fmtNum(v)}` });
      afterGap = false;
    });
    // shade the prompt and mask=0 runs: regions the policy did not generate
    const bands = [{ x0: 0, x1: payload.prompt_len, strong: true, label: "prompt" }];
    const mask = payload.loss_mask;
    if (mask) {
      let runStart = null;
      mask.forEach((m, i) => {
        if (m === 0 && runStart === null) runStart = i;
        if (m !== 0 && runStart !== null) {
          bands.push({ x0: payload.prompt_len + runStart, x1: payload.prompt_len + i });
          runStart = null;
        }
      });
      if (runStart !== null) bands.push({ x0: payload.prompt_len + runStart, x1: payload.prompt_len + mask.length });
    }
    queueMicrotask(() => drawChart(chartCanvas, points, { bands }));
  }

  // ---- token strip ----
  // the loss ignores masked positions, so they are dimmed and never tinted: a
  // color there would read as signal that does not exist
  const isMasked = (respPos) =>
    payload.loss_mask !== null && respPos >= 0 && respPos < payload.loss_mask.length && payload.loss_mask[respPos] === 0;
  const responseLen = (payload.train_log_probs ?? payload.rollout_log_probs ?? []).length;
  const heading = el("h3", {});
  // built once, then only recolored when "color by" changes: a rebuild would
  // throw away wherever in the sequence the reader had scrolled to
  const spans = payload.token_ids.map((tokenId, i) => {
    const respPos = i - payload.response_offset; // index into stat arrays
    const inResponse = respPos >= 0 && respPos < responseLen;
    const text = payload.token_text ? payload.token_text[i] : `·${tokenId}`;
    const masked = inResponse && isMasked(respPos);
    const span = el("span", { class: `tok ${inResponse ? "" : "prompt"} ${masked ? "masked" : ""}` }, [
      text.replaceAll("\n", "⏎\n"),
    ]);
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
  const box = el("div", { class: "tokens" }, spans);

  // the color scale spans the whole response, so a given value keeps one color
  // no matter where in the sequence it sits
  function paintStrip() {
    const values = payload[stat] ?? [];
    const color = values.length ? colorFor(stat, values) : () => "transparent";
    spans.forEach((span, i) => {
      const respPos = i - payload.response_offset;
      const value = respPos >= 0 && respPos < values.length ? values[respPos] : null;
      const tinted = value !== null && value !== undefined && !isMasked(respPos);
      span.style.background = tinted ? color(value) : "";
    });
    heading.textContent = `Tokens — colored by ${stat}${payload.token_text ? "" : " (no tokenizer: ids shown)"}`;
  }

  const statSelect = el("select", {}, available.map((s) =>
    Object.assign(el("option", { value: s }, [s]), { selected: s === stat }),
  ));
  statSelect.onchange = () => {
    stat = statSelect.value;
    paintStrip();
  };
  controls.replaceChildren(
    el("span", {}, [`${payload.total_len} tokens (prompt ${payload.prompt_len})`]),
    el("span", { class: "muted" }, [" color by"]),
    statSelect,
  );
  strip.replaceChildren(heading, box);
  paintStrip();

  root.replaceChildren(controls, strip, chartPanel);

  // Sizing the chart canvas and reading a token's offsetTop both need a layout
  // box, and this load is slow enough to carry its own "several minutes" notice
  // — long enough for the reader to have switched back to the Conversation tab
  // before it lands. Rendering into a hidden pane would silently produce a 0x0
  // canvas and a scrollTop of 0, so the layout-bound work waits to be replayed
  // by whoever shows the pane next.
  const firstResponse = spans[payload.response_offset];
  root._onShown = () => {
    if (!root.getClientRects().length) return; // still no layout box
    renderChart();
    // open on the first generated token: an agentic prompt runs to thousands of
    // tokens of chat history, and none of the per-token metrics are defined over it
    if (firstResponse) box.scrollTop = firstResponse.offsetTop;
    root._onShown = null; // one-shot, so later tab switches never yank the reader's scroll
  };
  root._onShown();
}
