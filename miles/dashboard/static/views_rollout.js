import { createAnatomy } from "./anatomy.js";
import { api } from "./api.js";
import { el, fmtNum } from "./app.js";
import { drawChart } from "./charts.js";

const DEFAULT_COLUMNS = [
  "sample_index",
  "group_index",
  "raw_reward",
  "reward",
  "response_length",
  "truncated",
  "versions",
  "turns",
  "tool_calls",
  "mean_abs_lp_diff",
  "mean_imp_ratio",
  "mean_entropy",
  "adv_mean",
  "non_generation_time",
];

// staleness span rendered as one sortable-ish string column; mixed-version
// samples are exactly what --use-tis corrects, so they must not blend in
function versionSpan(row) {
  if (row.weight_version === null || row.weight_version === undefined) return null;
  const lo = row.weight_version_min;
  return row.mixed_version ? `v${lo}–v${row.weight_version}` : `v${row.weight_version}`;
}

function sortableTable(rows, columns, { onRowClick, flagRow, sortState }) {
  const wrap = el("div", { class: "tablewrap" });
  const render = () => {
    const { column, desc } = sortState;
    const sorted = [...rows].sort((a, b) => {
      const [va, vb] = [a[column], b[column]];
      if (va === null) return 1;
      if (vb === null) return -1;
      return (va > vb ? 1 : va < vb ? -1 : 0) * (desc ? -1 : 1);
    });
    const head = el("tr", {}, columns.map((c) =>
      el("th", {
        class: c === column ? "sorted" : "",
        onclick: () => {
          sortState.desc = sortState.column === c ? !sortState.desc : true;
          sortState.column = c;
          render();
        },
      }, [c === column ? `${c} ${desc ? "▾" : "▴"}` : c]),
    ));
    const body = sorted.map((row) =>
      el("tr", {
        class: flagRow && flagRow(row) ? "flagged" : "",
        onclick: onRowClick ? () => onRowClick(row) : null,
      }, columns.map((c) => el("td", {}, [fmtNum(row[c])]))),
    );
    wrap.replaceChildren(el("table", { class: "data" }, [el("thead", {}, [head]), el("tbody", {}, body)]));
  };
  render();
  return wrap;
}

function statBox(label, value) {
  return el("div", { class: "stat" }, [el("div", { class: "v" }, [fmtNum(value)]), el("div", { class: "k" }, [label])]);
}

export async function renderRollout(view, meta, route) {
  const { rolloutId, evaluation } = route;
  const [summary, groups] = await Promise.all([
    api(`/api/rollout/${rolloutId}/summary`, { eval: evaluation }),
    api(`/api/rollout/${rolloutId}/groups`, { eval: evaluation }),
  ]);
  const rows = summary.rows;
  for (const row of rows) row.versions = versionSpan(row);
  // lifecycle lanes exist only for runs with the trajectory probes (PR §18);
  // older dumps or eval steps: no pane, table only
  let trajectories = { lanes: [], consume_ts: null };
  if (!evaluation) {
    try {
      trajectories = await api(`/api/rollout/${rolloutId}/trajectories`);
    } catch {
      /* endpoint absent or no events: keep the table-only layout */
    }
  }
  const rewardKey = evaluation ? "reward" : "raw_reward";

  // -------- header controls: prev/next step, train/eval toggle --------
  const ids = evaluation ? meta.rollout_ids.eval : meta.rollout_ids.train;
  const position = ids.indexOf(rolloutId);
  const goto = (id, toEval) => (location.hash = `#/rollout/${id}${toEval ? "?eval=1" : ""}`);
  // type a step number + Enter to jump straight to it
  const jumpInput = el("input", {
    type: "number",
    value: String(rolloutId),
    style: "width: 64px",
    onkeydown: (ev) => {
      if (ev.key === "Enter") goto(Number(ev.target.value), evaluation);
    },
  });
  const controls = el("div", { class: "controls" }, [
    el("button", { onclick: () => position > 0 && goto(ids[position - 1], evaluation) }, ["◀ prev"]),
    el("span", {}, [`${evaluation ? "eval" : "train"} step`]),
    jumpInput,
    el("span", {}, [`(${position + 1}/${ids.length})`]),
    el("button", { onclick: () => position < ids.length - 1 && goto(ids[position + 1], evaluation) }, ["next ▶"]),
  ]);
  if (!evaluation && meta.rollout_ids.eval.includes(rolloutId)) {
    controls.append(el("button", { onclick: () => goto(rolloutId, true) }, ["eval view"]));
  }
  if (evaluation && meta.rollout_ids.train.includes(rolloutId)) {
    controls.append(el("button", { onclick: () => goto(rolloutId, false) }, ["train view"]));
  }

  // -------- headline stats --------
  const rewards = rows.map((r) => r[rewardKey]).filter((v) => v !== null);
  const mean = (vs) => (vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null);
  const zeroStdGroups = groups.rows.filter((g) => g.zero_std).length;
  const stats = el("div", { class: "statgrid" }, [
    statBox("samples", rows.length),
    statBox("reward mean", mean(rewards)),
    statBox("truncated frac", mean(rows.map((r) => (r.truncated ? 1 : 0)))),
    statBox("zero-std groups", `${zeroStdGroups}/${groups.rows.length}`),
    statBox("mixed-version frac", mean(rows.map((r) => (r.mixed_version === null ? null : +r.mixed_version)).filter((v) => v !== null))),
    statBox("mean |lp diff|", mean(rows.map((r) => r.mean_abs_lp_diff).filter((v) => v !== null))),
    statBox("mean entropy", mean(rows.map((r) => r.mean_entropy).filter((v) => v !== null))),
  ]);

  // -------- tabs --------
  const openTokens = (row) =>
    (location.hash = `#/rollout/${rolloutId}/sample/${row.sample_index}${evaluation ? "?eval=1" : ""}`);

  const samplesTab = () => {
    const scatter = el("canvas", { class: "chart", style: "height: 260px" });
    queueMicrotask(() =>
      drawChart(
        scatter,
        rows
          .filter((r) => r[rewardKey] !== null)
          .map((r) => ({
            x: r.response_length,
            y: r[rewardKey],
            flag: Boolean(r.truncated),
            label: `sample ${r.sample_index}\nreward=${fmtNum(r[rewardKey])} len=${r.response_length}` +
              (r.mean_abs_lp_diff !== null ? `\n|lp diff|=${fmtNum(r.mean_abs_lp_diff)}` : ""),
            row: r,
          })),
        { line: false, onClick: (p) => openTokens(p.row) },
      ),
    );
    const allColumns = el("input", { type: "checkbox" });
    const tableHolder = el("div", {});
    const renderTable = () => {
      const columns = allColumns.checked
        ? summary.columns
        : DEFAULT_COLUMNS.filter((c) => summary.columns.includes(c));
      tableHolder.replaceChildren(
        sortableTable(rows, columns, {
          onRowClick: openTokens,
          flagRow: (r) => r.remove_sample,
          sortState: { column: "sample_index", desc: false },
        }),
      );
    };
    allColumns.onchange = renderTable;
    renderTable();
    const panels = [];
    if (trajectories.lanes.length) {
      panels.push(
        createAnatomy({
          lanes: trajectories.lanes,
          consumeTs: trajectories.consume_ts,
          rowsByIndex: new Map(rows.map((r) => [r.sample_index, r])),
          onClickSample: (index) => openTokens({ sample_index: index }),
        }),
      );
    }
    return [
      ...panels,
      el("div", { class: "panel" }, [el("h3", {}, [`${rewardKey} vs response length (red = truncated)`]), scatter]),
      el("div", { class: "panel" }, [
        el("h3", {}, ["samples — click a row for the token view"]),
        el("div", { class: "controls" }, [el("label", {}, [allColumns, " all columns"])]),
        tableHolder,
      ]),
    ];
  };

  const groupsTab = () => [
    el("div", { class: "panel" }, [
      el("h3", {}, ["GRPO groups (red = zero reward std → no gradient signal)"]),
      sortableTable(groups.rows, groups.columns, {
        flagRow: (g) => g.zero_std,
        sortState: { column: "group_index", desc: false },
      }),
    ]),
  ];

  const tabBody = el("div", {});
  const tabButtons = el("div", { class: "tabs" });
  const tabs = { samples: samplesTab, groups: groupsTab };
  const selectTab = (name) => {
    tabBody.replaceChildren(...tabs[name]());
    tabButtons.replaceChildren(
      ...Object.keys(tabs).map((t) =>
        el("button", { class: t === name ? "active" : "", onclick: () => selectTab(t) }, [t]),
      ),
    );
  };
  selectTab("samples");

  view.replaceChildren(controls, stats, tabButtons, tabBody);
}
