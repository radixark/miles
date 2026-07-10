import { api } from "./api.js";
import { el } from "./app.js";
import { drawChart } from "./charts.js";

const PINNED_STORE = ["rollout/rewards_mean", "train/loss", "perf/step_time", "perf/wait_time_ratio"];
const PINNED_DUMP = [
  "dump/reward_mean",
  "dump/mean_abs_lp_diff",
  "dump/mean_entropy",
  "dump/truncated_frac",
  "dump/zero_std_group_frac",
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
  const pinned = [...PINNED_STORE, ...PINNED_DUMP].filter((k) => meta.metric_keys.includes(k));
  return new Set(pinned.slice(0, 6));
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
          return el("label", {}, [box, ` ${key}`]);
        }),
    );
  };
  filterInput.oninput = renderKeyList;

  async function renderCharts() {
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
    const series = {};
    await Promise.all(
      [...byAxis.entries()].map(async ([axis, keys]) => {
        const got = await api("/api/metrics", {
          keys: keys.join(","),
          x: axis === "dump" ? "rollout/step" : axis,
        });
        Object.assign(series, got);
      }),
    );

    chartsPanel.replaceChildren(
      ...[...selected].map((key) => {
        const axis = axisOf(key);
        const canvas = el("canvas", { class: "chart" });
        const panel = el("div", { class: "panel" }, [el("p", { class: "chart-title" }, [key]), canvas]);
        queueMicrotask(() => {
          const s = series[key] ?? { x: [], y: [] };
          const stepNavigable = axis === "dump" || axis === "rollout/step";
          drawChart(
            canvas,
            s.x.map((x, i) => ({ x, y: s.y[i], label: `step ${x}\n${key} = ${s.y[i]}` })),
            {
              onClick: stepNavigable ? (p) => (location.hash = `#/rollout/${p.x}`) : null,
            },
          );
        });
        return panel;
      }),
    );
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
  await renderCharts();
}
