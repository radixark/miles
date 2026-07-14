import { getMeta } from "./api.js";
import { renderMetrics } from "./views_metrics.js";
import { renderRollout } from "./views_rollout.js";
import { renderTimeline } from "./views_timeline.js";
import { renderTokens } from "./views_tokens.js";

// tiny DOM builder: el("div", {class: "x", onclick: fn}, [children|strings])
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) node[k] = v;
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

// views with background work (follow-mode auto refresh) register a cleanup
// here; render() runs it before switching views so intervals never leak
let activeViewCleanup = null;

export function setViewCleanup(cleanup) {
  activeViewCleanup = cleanup;
}

export const fmtNum = (v) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "✓" : "";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1e5 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  return v.toFixed(3);
};

function parseRoute() {
  const [path, query] = (location.hash.slice(1) || "/").split("?");
  const segments = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  if (segments[0] === "timeline") {
    return { view: "timeline", lanes: params.get("lanes") };
  }
  if (segments[0] === "rollout" && segments.length >= 2) {
    const rolloutId = Number(segments[1]);
    const evaluation = params.get("eval") === "1";
    if (segments[2] === "sample" && segments.length === 4) {
      return { view: "tokens", rolloutId, sampleIndex: Number(segments[3]), evaluation };
    }
    return { view: "rollout", rolloutId, evaluation };
  }
  return { view: "metrics" };
}

function crumbs(route, meta) {
  const nav = (label, href, active) => el("a", { class: `nav${active ? " active" : ""}`, href }, [label]);
  const parts = [nav("Metrics", "#/", route.view === "metrics")];
  if (meta.capabilities.has_timeline) {
    parts.push(nav("Compute Utilization", "#/timeline", route.view === "timeline"));
  }
  // the per-step data view is a top-level destination, not a hidden
  // click-through from chart points; land on the newest train step
  const latest = meta.rollout_ids.train.at(-1);
  if (latest !== undefined) {
    parts.push(nav("Data / Trajectories", `#/rollout/${latest}`, route.view === "rollout" || route.view === "tokens"));
  }
  if (route.view === "rollout" || route.view === "tokens") {
    const evalSuffix = route.evaluation ? "?eval=1" : "";
    parts.push(
      el("span", { class: "crumb" }, [
        "› ",
        el("a", { href: `#/rollout/${route.rolloutId}${evalSuffix}` }, [
          `${route.evaluation ? "eval " : ""}step ${route.rolloutId}`,
        ]),
      ]),
    );
  }
  if (route.view === "tokens") {
    parts.push(el("span", { class: "crumb" }, [`› sample ${route.sampleIndex}`]));
  }
  document.getElementById("crumbs").replaceChildren(...parts);
}

async function render() {
  if (activeViewCleanup) {
    activeViewCleanup();
    activeViewCleanup = null;
  }
  const route = parseRoute();
  const view = document.getElementById("view");
  view.replaceChildren(el("p", { class: "muted" }, ["loading…"]));
  try {
    const meta = await getMeta();
    crumbs(route, meta);
    document.getElementById("runinfo").textContent =
      `${meta.run_name ?? "unnamed run"} · ${meta.mode}` + (meta.capabilities.has_metrics ? "" : " · dump-derived metrics");
    if (route.view === "metrics") await renderMetrics(view, meta);
    else if (route.view === "timeline") await renderTimeline(view, meta, route);
    else if (route.view === "rollout") await renderRollout(view, meta, route);
    else await renderTokens(view, meta, route);
  } catch (err) {
    view.replaceChildren(el("div", { class: "error" }, [String(err)]));
  }
}

window.addEventListener("hashchange", render);
render();
