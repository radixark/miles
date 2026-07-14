import { api } from "./api.js";
import { el, setViewCleanup } from "./app.js";
import { drawChart } from "./charts.js";

const PINNED_STORE = [
  "rollout/rewards_mean",
  "train/loss",
  "perf/step_time",
  "perf/wait_time_ratio",
  // staleness (fully async): logged by miles when weight versions appear
  "weight_version/mean",
  "weight_version/mixed_version_ratio",
];
const PINNED_DUMP = [
  "dump/reward_mean",
  "dump/mean_abs_lp_diff",
  "dump/mean_entropy",
  "dump/truncated_frac",
  "dump/zero_std_group_frac",
  "dump/mixed_version_frac",
];

function axisOf(key) {
  if (key.startsWith("dump/")) return "dump";
  if (key.startsWith("train/")) return "train/step";
  if (key.startsWith("eval/")) return "eval/step";
  return "rollout/step";
}

function selectedKeys(meta) {
  const saved = sessionStorage.getItem("selectedMetrics");
  if (saved !== null) {
    return new Set(JSON.parse(saved).filter((k) => meta.metric_keys.includes(k)));
  }
  // L0 defaults come from the telemetry stream only (the tracking-backend
  // fan-out, ms-cheap at any run length). dump/* series torch.load raw
  // sample dumps — a fallback for dump-only dirs and an explicit opt-in
  // otherwise, never part of the first paint.
  const pinned = meta.capabilities.has_metrics ? PINNED_STORE : PINNED_DUMP;
  return new Set(pinned.filter((k) => meta.metric_keys.includes(k)).slice(0, 6));
}

export async function renderMetrics(view, meta) {
  const selected = selectedKeys(meta);
  const chartsPanel = el("div", { style: "flex: 3; min-width: 500px" });

  const filterInput = el("input", { type: "text", placeholder: "filter keys…" });
  const keyList = el("div", { class: "keylist" });
  const renderKeyList = () => {
    const needle = filterInput.value.toLowerCase();
    keyList.replaceChildren(
      ...meta.metric_keys
        .filter((k) => k.toLowerCase().includes(needle))
        .map((key) => {
          const box = el("input", { type: "checkbox" });
          box.checked = selected.has(key);
          box.onchange = () => {
            box.checked ? selected.add(key) : selected.delete(key);
            sessionStorage.setItem("selectedMetrics", JSON.stringify([...selected]));
            renderCharts();
          };
          const slow = key.startsWith("dump/") && meta.capabilities.has_metrics;
          return el("label", {}, [box, ` ${key}`, ...(slow ? [el("span", { class: "muted" }, [" · slow: reads dumps"])] : [])]);
        }),
    );
  };
  filterInput.oninput = renderKeyList;

  function renderCharts() {
    if (!selected.size) {
      chartsPanel.replaceChildren(el("p", { class: "muted" }, ["select metrics on the left"]));
      return;
    }
    const byAxis = new Map();
    for (const key of selected) {
      const axis = axisOf(key);
      if (!byAxis.has(axis)) byAxis.set(axis, []);
      byAxis.get(axis).push(key);
    }
    // panels render immediately; each fills as its response arrives, so a
    // slow group (dump/* cold scans) only ever delays its own charts
    const slots = new Map();
    chartsPanel.replaceChildren(
      ...[...selected].map((key) => {
        const canvas = el("canvas", { class: "chart" });
        const status = el("p", { class: "muted" }, ["computing…"]);
        slots.set(key, { canvas, status });
        return el("div", { class: "panel" }, [el("p", { class: "chart-title" }, [key]), status, canvas]);
      }),
    );
    const epoch = (renderCharts.epoch = (renderCharts.epoch ?? 0) + 1);
    for (const [axis, keys] of byAxis.entries()) {
      api("/api/metrics", { keys: keys.join(","), x: axis === "dump" ? "rollout/step" : axis })
        .then((series) => {
          if (epoch !== renderCharts.epoch) return; // selection changed meanwhile
          for (const key of keys) {
            const slot = slots.get(key);
            if (!slot) continue;
            slot.status.remove();
            const s = series[key] ?? { x: [], y: [] };
            const stepNavigable = axis === "dump" || axis === "rollout/step";
            drawChart(
              slot.canvas,
              s.x.map((x, i) => ({ x, y: s.y[i], label: `step ${x}\n${key} = ${s.y[i]}` })),
              {
                onClick: stepNavigable ? (p) => (location.hash = `#/rollout/${p.x}`) : null,
              },
            );
          }
        })
        .catch((err) => {
          if (epoch !== renderCharts.epoch) return;
          for (const key of keys) {
            const slot = slots.get(key);
            if (slot) slot.status.textContent = String(err);
          }
        });
    }
  }

  view.replaceChildren(
    el("div", { class: "row" }, [
      el("div", { class: "panel", style: "flex: 1; min-width: 260px; max-width: 340px" }, [
        el("h3", {}, ["metrics"]),
        el("div", { class: "controls" }, [filterInput]),
        keyList,
      ]),
      chartsPanel,
    ]),
  );
  renderKeyList();
  renderCharts();

  if (meta.mode === "follow") {
    // renderCharts is fire-and-forget: failures land in per-panel status
    // and the epoch guard drops stale responses across ticks
    const intervalId = setInterval(renderCharts, 5000);
    setViewCleanup(() => clearInterval(intervalId));
  }
}
