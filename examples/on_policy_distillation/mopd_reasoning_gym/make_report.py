"""Build a self-contained HTML report page from analyze.py's outputs (optional tooling).

Writes one page next to the figures analyze.py saved, so the report directory can be shared as is.

Example:
    python examples/on_policy_distillation/mopd_reasoning_gym/make_report.py \
        --report /root/shared_data/mopd-report/<run_id> --out /root/shared_data/mopd-report/<run_id>/index.html \
        --hardware "8x GB300 (2 x 4-GPU trays)" --layout gb300-sync \
        --compare "sync=/root/shared_data/mopd-report/<sync run_id>"
"""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path

TEACHER = {"caesar_cipher": 0.733, "simple_geometry": 0.998}

STYLE = """
<title>MOPD Run Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one reading column for prose, full-width figure plates and tables; data in mono. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --plate: #fcfcfb; --fg: #15181d; --muted: #586070; --line: #dde1e8;
  --accent: #2a78d6; --accent-ink: #1c5cab; --warn: #b4560f; --crit: #c23434; --ok: #127a12;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #101317; --surface: #171b21; --plate: #fcfcfb; --fg: #e8ebf0; --muted: #a2aab7; --line: #2a3039;
  --accent: #5598e7; --accent-ink: #86b6ef; --warn: #ec9a5a; --crit: #e66767; --ok: #4cc04c; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #101317; --surface: #171b21; --plate: #fcfcfb; --fg: #e8ebf0; --muted: #a2aab7; --line: #2a3039;
  --accent: #5598e7; --accent-ink: #86b6ef; --warn: #ec9a5a; --crit: #e66767; --ok: #4cc04c; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
.wrap { max-width: 1040px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; display: grid; gap: 44px; }
header { display: grid; gap: 12px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); }
h1 { font: 700 clamp(30px, 5vw, 44px)/1.05 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 24px/1.2 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 600 17px/1.3 var(--display); margin: 0; }
p, li { max-width: 70ch; }
p { margin: 0; }
section { display: grid; gap: 16px; min-width: 0; }
.lede { font-size: 17px; color: var(--muted); max-width: 72ch; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font: 500 12px/1 var(--mono); padding: 7px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); background: var(--surface); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; display: grid; gap: 4px; align-content: start; }
.tile .k { font: 500 12px/1.3 var(--mono); color: var(--muted); letter-spacing: .02em; }
.tile .v { font: 600 28px/1.1 var(--display); font-variant-numeric: tabular-nums; }
.tile .s { font-size: 13px; color: var(--muted); }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font: 600 12px/1.3 var(--mono); color: var(--muted); letter-spacing: .03em; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.ours { background: color-mix(in srgb, var(--accent) 7%, transparent); }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
figure img { width: 100%; height: auto; background: var(--plate); border: 1px solid var(--line); border-radius: 10px; }
figcaption { font-size: 13px; color: var(--muted); max-width: 80ch; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 20px; }
.finding { display: grid; gap: 6px; padding-left: 14px; border-left: 3px solid var(--line); }
.finding.crit { border-left-color: var(--crit); }
.finding.warn { border-left-color: var(--warn); }
.sev { font: 600 11px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
.crit .sev { color: var(--crit); } .warn .sev { color: var(--warn); } .info .sev { color: var(--muted); }
code { font: 13px var(--mono); background: color-mix(in srgb, var(--muted) 12%, transparent); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
.note { font-size: 13px; color: var(--muted); }
@media (max-width: 560px) { .wrap { padding-block: 24px 48px; gap: 32px; } .tile .v { font-size: 24px; } }
</style>
"""


def _pct(x):
    return "–" if x is None else f"{x:.1%}"


def _num(x, digits=0):
    return "–" if x is None or x == "" else f"{float(x):,.{digits}f}"


def _run_stats(directory: Path):
    """Medians over updates 1-4 (past warm-up, before the first in-loop eval) plus final eval."""
    rows = [r for r in csv.DictReader((directory / "steps.csv").open()) if r.get("step_time")]
    window = [r for r in rows if 1 <= int(r["step"]) <= 4]

    def med(key):
        values = [float(r[key]) for r in window if r.get(key)]
        return statistics.median(values) if values else None

    summary = json.loads((directory / "summary.json").read_text())
    final = {t: (summary["eval"][t][-1] if summary["eval"][t] else None) for t in TEACHER}
    return {
        "step": med("step_time_excl_eval"), "rollout": med("rollout_time"), "p90": med("batch_p90_done_s"),
        "tail": med("longest_decode_tok_per_s"), "rollout_tps": med("rollout_tok_per_s_per_gpu"),
        "trainer_tps": med("trainer_tok_per_s_per_gpu"), "updates": len(rows), "final": final,
    }  # fmt: skip


def _compare_table(compare):
    head = (
        '<tr><th>run</th><th class="n">step s</th><th class="n">rollout s</th><th class="n">90% of batch s</th>'
        '<th class="n">tail decode tok/s</th><th class="n">rollout tok/s/GPU</th><th class="n">trainer tok/s/GPU</th>'
        '<th class="n">final caesar / geometry</th></tr>'
    )
    body = []
    for label, directory in compare:
        st = _run_stats(directory)
        final = " / ".join(
            f"{_pct(st['final'][t][1])} @{st['final'][t][0] + 1}" if st["final"][t] else "–" for t in TEACHER
        )
        body.append(
            f"<tr><td>{html.escape(label)}</td><td class=\"n\">{_num(st['step'])}</td><td class=\"n\">{_num(st['rollout'])}</td>"
            f"<td class=\"n\">{_num(st['p90'])}</td><td class=\"n\">{_num(st['tail'])}</td><td class=\"n\">{_num(st['rollout_tps'])}</td>"
            f"<td class=\"n\">{_num(st['trainer_tps'])}</td><td class=\"n\">{final}</td></tr>"
        )
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path, help="analyze.py output directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hardware", required=True, help='e.g. "16x B200 (2 x 8-GPU nodes)"')
    parser.add_argument("--layout", required=True, help="the launcher's --layout, or a description of the GPU split")
    parser.add_argument("--compare", nargs="*", default=[], help="LABEL=DIR of other analyze.py outputs")
    args = parser.parse_args()

    summary = json.loads((args.report / "summary.json").read_text())
    compare = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in args.compare]
    compare_html = (
        "<section><h2>Runs side by side</h2>"
        '<p class="note">Medians over updates 1–4 (past kernel warm-up, before the first in-loop eval); step time '
        "excludes eval. Final eval is avg@4 after the given number of updates.</p>"
        f"{_compare_table([('this run', args.report), *compare])}</section>"
        if compare
        else ""
    )
    rows = list(csv.DictReader((args.report / "steps.csv").open()))
    final = {t: (summary["eval"][t][-1][1] if summary["eval"][t] else None) for t in TEACHER}
    first = {t: (summary["eval"][t][0][1] if summary["eval"][t] else None) for t in TEACHER}
    steady = [r for r in rows if r.get("step_time_excl_eval") and int(r["step"]) >= 3]
    rollout_share = (
        sum(float(r["train_wait_time"]) - float(r.get("eval_s") or 0) for r in steady)
        / sum(float(r["step_time_excl_eval"]) for r in steady)
        if steady else None
    )  # fmt: skip
    done = summary["batch_done_s_median"]

    def tile(k, v, s):
        return (
            f'<div class="tile"><span class="k">{k}</span><span class="v">{v}</span><span class="s">{s}</span></div>'
        )

    tiles = "".join(
        [
            tile("caesar_cipher held-out", _pct(final["caesar_cipher"]),
                 f'from {_pct(first["caesar_cipher"])} · teacher {_pct(TEACHER["caesar_cipher"])}'),
            tile("simple_geometry held-out", _pct(final["simple_geometry"]),
                 f'from {_pct(first["simple_geometry"])} · teacher {_pct(TEACHER["simple_geometry"])}'),
            tile("median step, excl. eval", f'{_num(summary["step_time_excl_eval_s"]["median"])} s',
                 f'range {_num(summary["step_time_excl_eval_s"]["min"])}–{_num(summary["step_time_excl_eval_s"]["max"])} s'),
            tile("trainer waiting on rollouts", _pct(rollout_share), "share of step time, updates 3+"),
        ]
    )  # fmt: skip

    step_rows = "".join(
        "<tr>"
        + f'<td class="n">{r["step"]}</td>'
        + f'<td class="n">{_num(r.get("step_time_excl_eval"))}</td>'
        + f'<td class="n">{_num(float(r["train_wait_time"]) - float(r.get("eval_s") or 0)) if r.get("train_wait_time") else "–"}</td>'
        + f'<td class="n">{_num(r.get("log_probs_time"))}</td>'
        + f'<td class="n">{_num(r.get("actor_train_time"))}</td>'
        + f'<td class="n">{_num(r.get("eval_s"))}</td>'
        + f'<td class="n">{_num(r.get("batch_p50_done_s"))} / {_num(r.get("batch_p90_done_s"))} / {_num(r.get("batch_last_done_s"))}</td>'
        + f'<td class="n">{_num(r.get("max_response_len"))}</td>'
        + f'<td class="n">{_num(r.get("trainer_tok_per_s_per_gpu"))}</td>'
        + f'<td class="n">{_pct(float(r["actor_train_mfu"])) if r.get("actor_train_mfu") else "–"}</td>'
        + f'<td class="n">{_pct(float(r["raw_reward"])) if r.get("raw_reward") else "–"}</td>'
        + "</tr>"
        for r in rows
        if r.get("step_time")
    )

    page = f"""{STYLE}
<div class="wrap">
<header>
  <span class="eyebrow">Multi-teacher on-policy distillation · Miles</span>
  <h1>Qwen3.6-35B-A3B distilled from two reasoning_gym teachers</h1>
  <p class="lede">Student Qwen3.6-35B-A3B, one frozen GRPO teacher per task (caesar_cipher, simple_geometry), sampled-token reverse-KL OPD, {summary["steps_trained"]} updates × 128 prompts × 1 response, 30,720-token responses. Hardware: {html.escape(args.hardware)}. Layout: {html.escape(args.layout)}.</p>
  <div class="chips">
    <span class="chip">Qwen3.6-35B-A3B student</span><span class="chip">2 GRPO teachers, routed by task</span>
    <span class="chip">sampled-token reverse-KL OPD</span><span class="chip">30,720-token responses</span>
    <span class="chip">{summary["dropped_rollouts"]} rollouts dropped</span>
  </div>
</header>

<section>
  <h2>Result</h2>
  <div class="tiles">{tiles}</div>
  <p class="note">Held-out eval is avg@4 on 128 reasoning_gym prompts per task (seed 10000), exact match on the last <code>&lt;answer&gt;</code> after <code>&lt;/think&gt;</code>, the protocol on the teacher cards.</p>
</section>

<section>
  <h2>Where step time goes</h2>
  <div class="grid2">
    <figure><img src="fig_batch_completion.png" alt="Fraction of each batch scored over time, and time to finish 50, 90 and 100 percent of each batch"><figcaption>Half of every batch is scored by about {_num(done["p50"])} s and 90% by {_num(done["p90"])} s, and the step waits until {_num(done["last"])} s for the last sample (medians across updates).</figcaption></figure>
    <figure><img src="fig_step_time.png" alt="Stacked bars of step time per update"><figcaption>Step time per update, split into rollout wait, old-policy forward, training forward and backward, weight sync, and held-out eval (excluded from cost).</figcaption></figure>
  </div>
</section>

{compare_html}

<section>
  <h2>Plots</h2>
  <p class="note">Trainer MFU is 6 × 3B active × trained tokens ÷ (train time × trainer GPUs × peak dense BF16).</p>
  <figure><img src="fig_eval.png" alt="Held-out accuracy by update for both tasks with teacher reference lines"><figcaption>Student held-out accuracy against each teacher.</figcaption></figure>
  <figure><img src="fig_trainer.png" alt="Trainer tokens per second per GPU and MFU by update"><figcaption>Trainer throughput and MFU, train phase only.</figcaption></figure>
  <figure><img src="fig_rollout.png" alt="Rollout latency, response length and throughput by update"><figcaption>Rollout P50/P90 end-to-end latency and response length per task, and generation throughput.</figcaption></figure>
  <figure><img src="fig_teacher.png" alt="Teacher sequence length, scoring latency and throughput by update"><figcaption>Teacher sequence length, scoring latency per rollout, and tokens scored per GPU.</figcaption></figure>
</section>

<section>
  <h2>Per-step data</h2>
  <div class="scroll"><table>
    <thead><tr><th class="n">update</th><th class="n">step s (excl. eval)</th><th class="n">rollout wait s</th><th class="n">old-policy fwd s</th><th class="n">train fwd+bwd s</th><th class="n">eval s</th><th class="n">batch 50/90/100% done s</th><th class="n">longest response</th><th class="n">trainer tok/s/GPU</th><th class="n">MFU (Miles)</th><th class="n">train accuracy</th></tr></thead>
    <tbody>{step_rows}</tbody>
  </table></div>
  <p class="note">Totals: {summary["steps_trained"]} updates, {_num(summary["trained_tokens_M"], 1)}M trained tokens, {_num(summary["training_wall_s_excl_eval_and_startup"] / 60, 1)} min of training wall-clock excluding eval and startup, {_num(summary["gpu_hours"], 1)} GPU-hours on {summary["total_gpus"]} GPUs.</p>
</section>
</div>
"""
    args.out.write_text(page)
    print(args.out)


if __name__ == "__main__":
    main()
