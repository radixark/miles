import { api } from "./api.js";
import { el, setViewCleanup, statBox } from "./app.js";
import { drawMultiLine, SERIES_COLORS } from "./charts.js";

const REFRESH_MS = 5000;

const fmtAge = (seconds) => {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
};

export async function renderSandboxes(view, meta, route) {
  const staleMinutes = route.staleMinutes ?? sessionStorage.getItem("sandboxStaleMin") ?? "10";
  const staleInput = el("input", { type: "number", min: "1", value: staleMinutes });
  const stages = el("div", { class: "statgrid" });
  const failureTotals = el("div", { class: "statgrid" });
  const canvas = el("canvas", { class: "chart" });
  const staleTitle = el("h3", {}, ["Stale trials"]);
  const staleBody = el("div", { class: "tablewrap" });

  view.replaceChildren(
    el("div", { class: "controls" }, ["stale after (minutes)", staleInput]),
    el("div", { class: "panel" }, [el("h3", {}, ["Trials by current stage"]), stages]),
    el("div", { class: "panel" }, [el("h3", {}, ["Failures by exception type (per minute)"]), failureTotals, canvas]),
    el("div", { class: "panel" }, [staleTitle, staleBody]),
  );

  async function refresh() {
    const minutes = Math.max(1, Number(staleInput.value) || 10);
    sessionStorage.setItem("sandboxStaleMin", String(minutes));
    const status = await api("/api/sandboxes", { stale_after_s: minutes * 60, bucket_s: 60 });
    stages.replaceChildren(
      statBox("trials", status.num_trials),
      ...Object.entries(status.stage_counts).map(([stage, count]) => statBox(stage, count)),
    );
    // one box per chart line, in line order, with the line's color as the legend
    failureTotals.replaceChildren(
      ...(status.failure_series.length
        ? status.failure_series.map((series, i) => {
            const box = statBox(series.label, status.failure_counts[series.label]);
            box.style.borderLeft = `4px solid ${SERIES_COLORS[i % SERIES_COLORS.length]}`;
            return box;
          })
        : [el("span", { class: "muted" }, ["no failures"])]),
    );
    drawMultiLine(canvas, status.failure_series);
    staleTitle.textContent = `Stale trials (${status.stale.length}) — no event for over ${minutes} min`;
    const header = ["trial", "task", "stage", "provider", "sandbox", "silent for"];
    staleBody.replaceChildren(
      status.stale.length
        ? el("table", { class: "data" }, [
            el("thead", {}, [el("tr", {}, header.map((h) => el("th", {}, [h])))]),
            el(
              "tbody",
              {},
              status.stale.map((row) =>
                el("tr", { class: "flagged" }, [
                  row.trial_id, row.task, row.stage, row.provider ?? "—", row.sandbox_id ?? "—", fmtAge(row.silent_s),
                ].map((cell) => el("td", {}, [cell]))),
              ),
            ),
          ])
        : el("p", { class: "muted" }, ["none"]),
    );
  }

  staleInput.onchange = refresh;
  await refresh();
  if (meta.mode === "follow") {
    const timer = setInterval(() => refresh().catch(() => {}), REFRESH_MS);
    setViewCleanup(() => clearInterval(timer));
  }
}
