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

export function statBox(label, value) {
  return el("div", { class: "stat" }, [el("div", { class: "v" }, [fmtNum(value)]), el("div", { class: "k" }, [label])]);
}

function parseRoute() {
  const [path, query] = (location.hash.slice(1) || "/").split("?");
  const segments = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  if (segments[0] === "timeline") {
    return { view: "timeline", lanes: params.get("lanes") };
  }
  if (segments[0] === "rollout" && segments.length >= 2) {
    const evaluation = params.get("eval") === "1";
    // "latest" is resolved against the dump at render time rather than baked
    // into the link, because the newest step is often listed before it is
    // readable; a typed step number is always honoured exactly
    if (segments[1] === "latest") {
      return { view: "rollout", rolloutId: null, evaluation };
    }
    const rolloutId = Number(segments[1]);
    if (segments[2] === "sample" && segments.length === 4) {
      return {
        view: "tokens",
        rolloutId,
        sampleIndex: Number(segments[3]),
        sampleOccurrence: Number(params.get("occurrence") || 0),
        evaluation,
      };
    }
    return { view: "rollout", rolloutId, evaluation };
  }
  return { view: "metrics" };
}

function crumbs(route, meta) {
  const nav = (label, href, active, onclick = null) =>
    el("a", { class: `nav${active ? " active" : ""}`, href, onclick }, [label]);
  const parts = [nav("Metrics", "#/", route.view === "metrics")];
  if (meta.capabilities.has_timeline) {
    parts.push(nav("Compute Utilization", "#/timeline", route.view === "timeline"));
  }
  // the per-step data view is a top-level destination, not a hidden
  // click-through from chart points; land on the newest usable train step
  if (meta.rollout_ids.train.length) {
    const onRollout = route.view === "rollout" || route.view === "tokens";
    const replaceInPlace = (event) => {
      event.preventDefault();
      location.replace("#/rollout/latest");
    };
    parts.push(nav("Rollouts", "#/rollout/latest", onRollout, onRollout ? replaceInPlace : null));
  }
  if ((route.view === "rollout" || route.view === "tokens") && route.rolloutId !== null) {
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
    const occurrence = route.sampleOccurrence ? ` · leaf ${route.sampleOccurrence + 1}` : "";
    parts.push(el("span", { class: "crumb" }, [`› sample ${route.sampleIndex}${occurrence}`]));
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
    const modeLabel =
      meta.mode === "follow" ? "● live — auto-refreshing from a still-running job" : "◼ static snapshot — job has finished";
    const runinfo = [
      el("span", {}, [`run: ${meta.run_name ?? "unnamed run"}`]),
      " · ",
      el("span", { title: modeLabel }, [meta.mode === "follow" ? "● live" : "◼ static"]),
      ...(meta.capabilities.has_metrics ? [] : [" · dump-derived metrics"]),
    ];
    if (meta.wandb_url) runinfo.push(" · ", el("a", { href: meta.wandb_url, target: "_blank" }, ["wandb ↗"]));
    document.getElementById("runinfo").replaceChildren(...runinfo);
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
