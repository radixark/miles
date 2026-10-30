import { api } from "./api.js";
import { el, setViewCleanup, statBox } from "./app.js";

const REFRESH_MS = 5000;

const fmtAgo = (seconds) => (seconds < 60 ? `${Math.round(seconds)} s ago` : `${Math.round(seconds / 60)} min ago`);

export async function renderSandboxes(view, meta, route) {
  const staleMinutes = route.staleMinutes ?? sessionStorage.getItem("sandboxStaleMin") ?? "5";
  const staleInput = el("input", { type: "number", min: "1", value: staleMinutes });
  const counts = el("div", { class: "statgrid" });
  const table = el("div", { class: "tablewrap" });
  view.replaceChildren(
    el("div", { class: "controls" }, ["stale after (minutes)", staleInput]),
    el("div", { class: "panel" }, [counts, table]),
  );

  async function refresh() {
    const minutes = Math.max(1, Number(staleInput.value) || 5);
    sessionStorage.setItem("sandboxStaleMin", String(minutes));
    const status = await api("/api/sandboxes", { stale_after_s: minutes * 60 });
    counts.replaceChildren(
      ...["healthy", "error", "stale", "finished"].map((name) => statBox(name, status.counts[name])),
    );
    const header = ["sandbox", "task", "provider", "health", "last error", "last executed", "stale"];
    table.replaceChildren(
      status.sandboxes.length
        ? el("table", { class: "data" }, [
            el("thead", {}, [el("tr", {}, header.map((h) => el("th", {}, [h])))]),
            el(
              "tbody",
              {},
              status.sandboxes.map((row) =>
                el(
                  "tr",
                  row.status === "error" || row.stale ? { class: "flagged" } : {},
                  [
                    row.sandbox_id ? `${row.sandbox_key} (${row.sandbox_id})` : row.sandbox_key,
                    row.task ?? "—",
                    row.provider ?? "—",
                    row.status,
                    row.last_error ?? "—",
                    fmtAgo(row.age_s),
                    row.stale ? "stale" : "",
                  ].map((cell) => el("td", {}, [cell])),
                ),
              ),
            ),
          ])
        : el("p", { class: "muted" }, ["no sandbox has reported yet"]),
    );
  }

  staleInput.onchange = refresh;
  await refresh();
  if (meta.mode === "follow") {
    const timer = setInterval(() => refresh().catch(() => {}), REFRESH_MS);
    setViewCleanup(() => clearInterval(timer));
  }
}
