import { api } from "./api.js";
import { el, setViewCleanup } from "./app.js";
import { drawChart, drawMultiLine, SERIES_COLORS } from "./charts.js";

// wandb-shaped L0: metric names come verbatim from the tracking fan-out, so
// categories are simply the key prefixes. One category at a time (wandb
// sidebar idiom); every key of the active category renders as a line chart.
// "dump" only ever appears for dump-only dirs — the server does not
// advertise it while a telemetry stream exists.
const CATEGORY_ORDER = ["rollout", "perf", "train", "eval", "dump"];

// trainer-side recomputations logged under the rollout/step axis (they share
// that step key, not because they're rollout-generation metrics) — displayed
// under "train" so they don't get lost among reward/response_length. Display
// grouping only: the key name and its rollout/step x-axis are untouched.
const TRAIN_DISPLAY_OVERRIDE = new Set([
  "rollout/log_probs",
  "rollout/ref_log_probs",
  "rollout/rollout_log_probs",
  "rollout/teacher_log_probs",
]);

const categoryOf = (key) => (TRAIN_DISPLAY_OVERRIDE.has(key) ? "train" : key.split("/")[0]);

// one-line descriptions shown as a native tooltip on the chart title, matched
// by substring against the part after the category prefix — most specific
// pattern first, so e.g. "ref_log_probs" is checked before generic
// "log_probs". Best-effort v1; extend as new metrics show up.
const METRIC_DESCRIPTIONS = [
  ["teacher_log_probs", "log-prob under a distillation teacher model"],
  ["ref_log_probs", "log-prob under the reference (KL-anchor) policy"],
  ["rollout_log_probs", "log-prob reported by the rollout engine at generation time"],
  ["log_probs", "trainer-recomputed log-prob of the response under the current policy"],
  ["ref_entropy", "per-token entropy under the reference policy"],
  ["entropy", "trainer-recomputed per-token entropy of the response distribution"],
  ["response_length", "generated-response length in tokens"],
  ["reward", "scalar reward assigned to the sampled response"],
  ["truncated", "samples that hit the length/turn limit before finishing"],
  ["zero_std", "GRPO groups whose reward std is ~0 — no gradient signal from that group"],
  ["mean_abs_lp_diff", "mean |trainer log-prob − rollout log-prob| — rollout/train policy staleness"],
  ["mixed_version", "samples whose tokens span more than one weight version"],
  ["step_time", "wall-clock seconds for one step"],
  ["wait_time_ratio", "fraction of the step spent idle/waiting rather than computing"],
  ["cpu_memory", "host CPU memory (RSS) sampled around this step, in GB"],
  ["advantages", "GRPO advantage (reward normalized within its group)"],
  ["returns", "return used for the policy-gradient / value target"],
  ["grad_norm", "global gradient norm before clipping"],
  ["pg_clipfrac", "fraction of tokens where the PPO clip bound was active"],
  ["kl", "KL divergence between the current and reference policy"],
  ["loss", "training loss for this step"],
  ["num_running_reqs", "requests concurrently running on the sglang engine"],
  ["gen_throughput", "generated tokens per second"],
  ["token_usage", "fraction of KV cache capacity currently in use"],
  ["cache_hit_rate", "prefix-cache hit rate"],
];

function describeMetric(key) {
  const suffix = key.slice(key.indexOf("/") + 1);
  const hit = METRIC_DESCRIPTIONS.find(([pattern]) => suffix.includes(pattern));
  return hit ? hit[1] : undefined;
}

function axisOf(key) {
  if (key.startsWith("dump/")) return "dump";
  if (key.startsWith("train/")) return "train/step";
  if (key.startsWith("eval/")) return "eval/step";
  return "rollout/step";
}

function orderedCategories(meta) {
  const present = [...new Set(meta.metric_keys.map(categoryOf))];
  const cats = [
    ...CATEGORY_ORDER.filter((c) => present.includes(c)),
    ...present.filter((c) => !CATEGORY_ORDER.includes(c) && c !== "dump").sort(),
  ];
  // grafana-parity category: scraped engine metrics, x = wall clock
  if (meta.engine_metric_keys?.length) cats.splice(cats.includes("dump") ? cats.length - 1 : cats.length, 0, "sglang");
  return cats;
}

export async function renderMetrics(view, meta) {
  const cats = orderedCategories(meta);
  let active = sessionStorage.getItem("metricsCategory");
  if (!cats.includes(active)) active = cats[0] ?? null;

  const chartsPanel = el("div", {
    style: "flex: 3; min-width: 500px; display: grid; grid-template-columns: repeat(2, minmax(0, 500px)); gap: 0 14px; align-content: start",
  });
  const filterInput = el("input", { type: "text", placeholder: "filter…" });
  const catList = el("div", { class: "keylist" });

  const renderCategories = () => {
    catList.replaceChildren(
      ...cats.map((cat) =>
        el(
          "button",
          {
            class: cat === active ? "active" : "",
            style: "text-align: left",
            onclick: () => {
              active = cat;
              sessionStorage.setItem("metricsCategory", cat);
              renderCategories();
              buildPanels();
            },
          },
          [cat],
        ),
      ),
    );
  };

  const activeKeys = () => {
    const needle = filterInput.value.toLowerCase();
    const pool = active === "sglang" ? meta.engine_metric_keys : meta.metric_keys.filter((k) => categoryOf(k) === active);
    return pool.filter((k) => k.toLowerCase().includes(needle));
  };

  // slots persist across refreshes; a chart repaints ONLY when its data
  // changed (signature check), so a live page sits visually still between
  // real updates instead of flickering on every poll
  const slots = new Map(); // key -> {canvas, status, signature, lastSeries?}
  let epoch = 0;

  // sglang legend: one checkbox per engine, all on by default; unchecking
  // hides that engine's line in every chart (and drops it from the y-scale)
  const engines = []; // sorted union of scraped addrs, fixes the color order page-wide
  const hiddenEngines = new Set();
  const legendPanel = el("div", { class: "panel", style: "flex: 0 0 240px; min-width: 240px; display: none" });
  const engineOpts = () => ({ hidden: hiddenEngines, colorIndex: (label) => engines.indexOf(label) });
  const redrawEngineCharts = () => {
    for (const slot of slots.values()) {
      if (slot.lastSeries) drawMultiLine(slot.canvas, slot.lastSeries, engineOpts());
    }
  };
  const renderLegend = () => {
    legendPanel.replaceChildren(
      el("h3", {}, ["Engines"]),
      ...engines.map((addr, i) =>
        el(
          "label",
          { style: "display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; font-size: 12px" },
          [
            el("input", {
              type: "checkbox",
              ...(hiddenEngines.has(addr) ? {} : { checked: "" }),
              onchange: (ev) => {
                if (ev.target.checked) hiddenEngines.delete(addr);
                else hiddenEngines.add(addr);
                redrawEngineCharts();
              },
            }),
            el("span", {
              style: `display: inline-block; width: 14px; height: 3px; background: ${SERIES_COLORS[i % SERIES_COLORS.length]}`,
            }),
            addr.replace(/^https?:\/\//, ""),
          ],
        ),
      ),
    );
  };

  function buildPanels() {
    slots.clear();
    legendPanel.style.display = active === "sglang" ? "" : "none";
    const keys = activeKeys();
    if (!keys.length) {
      chartsPanel.replaceChildren(el("p", { class: "muted" }, [active ? "no metrics here" : "no metrics logged"]));
      return;
    }
    chartsPanel.replaceChildren(
      ...keys.map((key) => {
        const canvas = el("canvas", { class: "chart" });
        const status = el("p", { class: "muted" }, ["loading…"]);
        slots.set(key, { canvas, status, signature: null });
        const title = el("p", { class: "chart-title", title: describeMetric(key) }, [key]);
        return el("div", { class: "panel" }, [title, status, canvas]);
      }),
    );
    refresh();
  }

  function refresh() {
    const current = ++epoch;
    if (active === "sglang") {
      for (const [key, slot] of slots.entries()) {
        api("/api/timeline/engine_series", { metric: key, max_points: 1000 })
          .then(({ series }) => {
            if (current !== epoch) return;
            const signature = series.map((s) => `${s.addr}:${s.ts.length}:${s.ts.at(-1)}`).join("|");
            if (signature === slot.signature) return;
            slot.signature = signature;
            slot.status.remove();
            slot.lastSeries = series.map((s) => ({ label: s.addr, ts: s.ts, value: s.value }));
            let grew = false;
            for (const s of series) {
              if (!engines.includes(s.addr)) {
                engines.push(s.addr);
                grew = true;
              }
            }
            if (grew) {
              engines.sort();
              renderLegend();
              redrawEngineCharts(); // sort may have shifted the color order
            } else {
              drawMultiLine(slot.canvas, slot.lastSeries, engineOpts());
            }
          })
          .catch((err) => {
            if (current === epoch && slot.signature === null) slot.status.textContent = String(err);
          });
      }
      return;
    }
    const byAxis = new Map();
    for (const key of slots.keys()) {
      const axis = axisOf(key);
      if (!byAxis.has(axis)) byAxis.set(axis, []);
      byAxis.get(axis).push(key);
    }
    for (const [axis, keys] of byAxis.entries()) {
      api("/api/metrics", { keys: keys.join(","), x: axis === "dump" ? "rollout/step" : axis })
        .then((series) => {
          if (current !== epoch) return; // category/filter changed meanwhile
          for (const key of keys) {
            const slot = slots.get(key);
            if (!slot) continue;
            const s = series[key] ?? { x: [], y: [] };
            const signature = `${s.x.length}:${s.x.at(-1)}:${s.y.at(-1)}`;
            if (signature === slot.signature) continue; // unchanged: no repaint
            slot.signature = signature;
            slot.status.remove();
            drawChart(
              slot.canvas,
              s.x.map((x, i) => ({ x, y: s.y[i], label: `step ${x}\n${key} = ${s.y[i]}` })),
            );
          }
        })
        .catch((err) => {
          if (current !== epoch) return;
          for (const key of keys) {
            const slot = slots.get(key);
            if (slot && slot.signature === null) slot.status.textContent = String(err);
          }
        });
    }
  }

  filterInput.oninput = buildPanels;

  view.replaceChildren(
    el("div", { class: "row" }, [
      el("div", { class: "panel", style: "flex: 0 0 200px; min-width: 200px" }, [
        el("h3", {}, ["Metrics"]),
        el("div", { class: "controls" }, [filterInput]),
        catList,
      ]),
      chartsPanel,
      legendPanel,
    ]),
  );
  renderCategories();
  buildPanels();

  if (meta.mode === "follow") {
    // refresh is fire-and-forget and idempotent: unchanged charts skip the
    // repaint entirely, so the live page sits still between real updates
    const intervalId = setInterval(refresh, 5000);
    setViewCleanup(() => clearInterval(intervalId));
  }
}
