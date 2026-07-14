// Minimal dependency-free canvas charts (line + scatter) with point picking.

const MARGIN = { left: 52, right: 14, top: 10, bottom: 24 };

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, width: rect.width, height: rect.height };
}

function niceTicks(min, max, target = 5) {
  if (!(max > min)) {
    max = min + (Math.abs(min) || 1);
  }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / target)));
  const step = [1, 2, 5, 10].map((m) => m * step0).find((s) => span / s <= target) || step0 * 10;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9 * span; t += step) ticks.push(t);
  return ticks;
}

const fmt = (v) => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return String(Math.round(v * 100) / 100);
  return String(Math.round(v * 1000) / 1000);
};

// points: [{x, y, label?, flag?}]; opts: {line: bool, onClick(point), color}
export function drawChart(canvas, points, opts = {}) {
  const { ctx, width, height } = setupCanvas(canvas);
  const css = getComputedStyle(document.documentElement);
  const colText = css.getPropertyValue("--muted").trim();
  const colBorder = css.getPropertyValue("--border").trim();
  const colMain = opts.color || css.getPropertyValue("--accent").trim();
  const colFlag = css.getPropertyValue("--bad").trim();
  ctx.clearRect(0, 0, width, height);

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  ctx.font = "11px ui-monospace, monospace";
  if (!points.length) {
    ctx.fillStyle = colText;
    ctx.fillText("no data", MARGIN.left + plotW / 2 - 20, MARGIN.top + plotH / 2);
    return;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let [xMin, xMax] = [Math.min(...xs), Math.max(...xs)];
  let [yMin, yMax] = [Math.min(...ys), Math.max(...ys)];
  if (xMin === xMax) [xMin, xMax] = [xMin - 0.5, xMax + 0.5];
  if (yMin === yMax) [yMin, yMax] = [yMin - 0.5, yMax + 0.5];
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;
  const X = (v) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const Y = (v) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = colBorder;
  ctx.fillStyle = colText;
  ctx.lineWidth = 1;
  for (const t of niceTicks(yMin, yMax)) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, Y(t));
    ctx.lineTo(width - MARGIN.right, Y(t));
    ctx.stroke();
    ctx.fillText(fmt(t), 4, Y(t) + 3);
  }
  for (const t of niceTicks(xMin, xMax, 8)) {
    ctx.fillText(fmt(t), X(t) - 8, height - 6);
  }

  if (opts.line !== false) {
    ctx.strokeStyle = colMain;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.stroke();
  }
  for (const p of points) {
    ctx.fillStyle = p.flag ? colFlag : colMain;
    ctx.beginPath();
    ctx.arc(X(p.x), Y(p.y), opts.line !== false ? 3 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  const nearest = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    let best = null;
    let bestDist = 20;
    for (const p of points) {
      const d = Math.hypot(X(p.x) - mx, Y(p.y) - my);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  };
  canvas.onmousemove = (ev) => {
    const p = nearest(ev);
    canvas.style.cursor = p && opts.onClick ? "pointer" : "default";
    if (p) {
      showTooltip(ev.clientX, ev.clientY, p.label ?? `${fmt(p.x)}, ${fmt(p.y)}`);
    } else {
      hideTooltip();
    }
  };
  canvas.onmouseleave = hideTooltip;
  if (opts.onClick) {
    canvas.onclick = (ev) => {
      const p = nearest(ev);
      if (p) opts.onClick(p);
    };
  }
}

// ------------------------------ tooltip -------------------------------------

let tooltipEl = null;

export function showTooltip(clientX, clientY, text) {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.id = "tooltip";
    document.body.appendChild(tooltipEl);
  }
  tooltipEl.textContent = text;
  tooltipEl.style.display = "block";
  const pad = 12;
  const w = tooltipEl.offsetWidth;
  const x = clientX + pad + w > window.innerWidth ? clientX - w - pad : clientX + pad;
  tooltipEl.style.left = `${x}px`;
  tooltipEl.style.top = `${Math.min(clientY + pad, window.innerHeight - tooltipEl.offsetHeight - pad)}px`;
}

export function hideTooltip() {
  if (tooltipEl) tooltipEl.style.display = "none";
}

// diverging color for values around a center (e.g. imp_ratio around 1)
export function divergingColor(t) {
  // t in [-1, 1]: blue -> transparent -> red
  const a = Math.min(1, Math.abs(t)) * 0.85;
  return t >= 0 ? `rgba(224, 96, 96, ${a})` : `rgba(78, 161, 255, ${a})`;
}

// sequential color for magnitudes in [0, 1]
export function sequentialColor(t) {
  return `rgba(70, 194, 142, ${Math.min(1, Math.max(0, t)) * 0.85})`;
}


// seriesList: [{label, ts: [], value: []}] — one thin line per engine on a
// shared scale; identity via hover (30+ engines make per-line color noise)
export function drawMultiLine(canvas, seriesList, opts = {}) {
  const { ctx, width, height } = setupCanvas(canvas);
  const css = getComputedStyle(document.documentElement);
  const colText = css.getPropertyValue("--muted").trim();
  const colBorder = css.getPropertyValue("--border").trim();
  const colMain = css.getPropertyValue("--accent").trim();
  ctx.clearRect(0, 0, width, height);
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  ctx.font = "11px ui-monospace, monospace";

  const alive = seriesList.filter((s) => s.ts.length);
  if (!alive.length) {
    ctx.fillStyle = colText;
    ctx.fillText("no data", MARGIN.left + plotW / 2 - 20, MARGIN.top + plotH / 2);
    return;
  }
  const xMin = Math.min(...alive.map((s) => s.ts[0]));
  const xMax = Math.max(...alive.map((s) => s.ts.at(-1)));
  let yMin = Math.min(...alive.map((s) => Math.min(...s.value)));
  let yMax = Math.max(...alive.map((s) => Math.max(...s.value)));
  if (yMin === yMax) [yMin, yMax] = [yMin - 0.5, yMax + 0.5];
  const X = (t) => MARGIN.left + ((t - xMin) / Math.max(xMax - xMin, 1e-9)) * plotW;
  const Y = (v) => MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = colBorder;
  ctx.fillStyle = colText;
  for (const tick of niceTicks(yMin, yMax, 4)) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, Y(tick));
    ctx.lineTo(width - MARGIN.right, Y(tick));
    ctx.stroke();
    ctx.fillText(fmt(tick), 6, Y(tick) + 4);
  }
  for (const tick of niceTicks(0, xMax - xMin, 6)) {
    const rel = Math.round(tick);
    ctx.fillText(`+${Math.floor(rel / 60)}:${String(rel % 60).padStart(2, "0")}`, X(xMin + tick) - 12, height - 6);
  }

  ctx.strokeStyle = colMain;
  ctx.lineWidth = 1;
  ctx.globalAlpha = Math.max(0.25, Math.min(1, 4 / alive.length));
  for (const s of alive) {
    ctx.beginPath();
    s.ts.forEach((t, i) => (i ? ctx.lineTo(X(t), Y(s.value[i])) : ctx.moveTo(X(t), Y(s.value[i]))));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    let best = null;
    for (const s of alive) {
      for (let i = 0; i < s.ts.length; i++) {
        const d = (X(s.ts[i]) - mx) ** 2 + (Y(s.value[i]) - my) ** 2;
        if (!best || d < best.d) best = { d, s, i };
      }
    }
    if (!best || best.d > 30 ** 2) {
      hideTooltip();
      return;
    }
    showTooltip(ev.clientX, ev.clientY, `${best.s.label}\n+${fmt(best.s.ts[best.i] - xMin)}s = ${fmt(best.s.value[best.i])}`);
  };
  canvas.onmouseleave = hideTooltip;
}
