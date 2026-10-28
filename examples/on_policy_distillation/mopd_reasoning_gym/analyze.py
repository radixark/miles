"""Turn a run's --dump-details directory into per-step tables and plots (optional tooling).

Needs matplotlib, which the Miles image does not require.

Reads:
    {dump}/dashboard/metrics.jsonl    every metric Miles logged, per step
    {dump}/rollout_data/{k}.pt        every training sample of step k, with the hooks.py timestamps

Writes to --out:
    steps.csv, summary.json           one row per step / run totals
    fig_eval.png                      student held-out accuracy vs teachers
    fig_step_time.png                 step time split into rollout, old-logprob forward, train, weight sync
    fig_trainer.png                   trainer tok/s/GPU and MFU
    fig_rollout.png                   P50/P90 rollout latency and length per task, rollout tok/s/GPU
    fig_teacher.png                   per-teacher P50 length, scoring latency, tok/s/GPU
    fig_batch_completion.png          when each sample of a batch finished: the tail that sets step time

Trainer MFU is 6 * N_active * trained tokens / (train time * trainer GPUs * peak dense BF16), with trained
tokens = prompt + response of every trained sample, so it is comparable across frameworks that log tokens.

Example (gb300-sync: 4 shared GPUs, 2 per teacher):
    python examples/on_policy_distillation/mopd_reasoning_gym/analyze.py \
        --dump /root/shared_data/mopd-dump/<run_id> --out /root/shared_data/mopd-report/<run_id> \
        --trainer-gpus 4 --rollout-gpus 4 --colocated --teacher-gpus caesar_cipher=2 simple_geometry=2 \
        --peak-tflops 2500
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TASKS = ("caesar_cipher", "simple_geometry")
COLOR = {"caesar_cipher": "#2a78d6", "simple_geometry": "#eb6834", "all": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Teacher held-out accuracy from the model cards: avg@4, seed 10000, 32k context, thinking on.
TEACHER_ACC = {"caesar_cipher": 0.733, "simple_geometry": 0.998}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
    }
)


def _metrics_by_step(dump: Path):
    steps = defaultdict(dict)
    for line in (dump / "dashboard" / "metrics.jsonl").open():
        record = json.loads(line)
        if record.get("step") is not None:
            steps[int(record["step"])].update(record["metrics"])
            if "perf/step_time" in record["metrics"]:
                steps[int(record["step"])]["_step_end_ts"] = record["ts"]
    return steps


def _eval_windows(dump: Path):
    """Wall-clock [start, end] of each in-loop eval, from the generation stamps of its samples."""
    windows = []
    for path in sorted((dump / "rollout_data").glob("*eval*.pt")):
        samples = torch.load(path, weights_only=False)["samples"]
        stamps = [s["metadata"] for s in samples if "gen_t0" in s["metadata"]]
        if stamps:
            windows.append((min(m["gen_t0"] for m in stamps), max(m["gen_t1"] for m in stamps)))
    return windows


def _samples_by_step(dump: Path):
    out = {}
    for path in sorted((dump / "rollout_data").glob("*.pt")):
        if re.fullmatch(r"\d+", path.stem):
            out[int(path.stem)] = torch.load(path, weights_only=False)["samples"]
    return out


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def _step_row(step, metrics, samples, args, teacher_gpus, eval_windows):
    row = {"step": step}
    for key in (
        "perf/step_time", "perf/rollout_time", "perf/train_wait_time", "perf/train_time", "perf/log_probs_time",
        "perf/actor_train_time", "perf/update_weights_time", "perf/sleep_time", "perf/wake_up_time",
        "perf/actor_train_mfu", "perf/actor_train_tok_per_s", "rollout/raw_reward", "train/opd_reverse_kl",
        "train/train_rollout_logprob_abs_diff", "train/grad_norm",
    ):  # fmt: skip
        row[key.split("/", 1)[1]] = metrics.get(key)
    for task in TASKS:
        row[f"eval_{task}"] = metrics.get(f"eval/{task}")
    row["eval_s"] = 0.0
    if row["step_time"] and "_step_end_ts" in metrics:
        end = metrics["_step_end_ts"]
        start = end - row["step_time"]
        row["eval_s"] = sum(max(0.0, min(end, t1) - max(start, t0)) for t0, t1 in eval_windows)
        row["step_time_excl_eval"] = row["step_time"] - row["eval_s"]
    if not samples:
        return row, None

    start = min(s["metadata"]["gen_t0"] for s in samples)
    trained_tokens = sum(len(s["tokens"]) for s in samples)
    row["trained_tokens"] = trained_tokens
    row["generated_tokens"] = sum(s["response_length"] for s in samples)
    finish = sorted(s["metadata"]["opd_score_t1"] - start for s in samples)
    row["batch_p50_done_s"], row["batch_p90_done_s"], row["batch_last_done_s"] = (
        _pct(finish, 50), _pct(finish, 90), finish[-1],
    )  # fmt: skip
    longest = max(samples, key=lambda s: s["response_length"])
    row["max_response_len"] = longest["response_length"]
    row["longest_decode_tok_per_s"] = longest["response_length"] / (
        longest["metadata"]["gen_t1"] - longest["metadata"]["gen_t0"]
    )
    rollout_time = row["rollout_time"] or finish[-1]
    row["rollout_tok_per_s_per_gpu"] = row["generated_tokens"] / rollout_time / args.rollout_gpus

    for task in TASKS:
        part = [s for s in samples if s["metadata"]["task"] == task]
        gen = [s["metadata"]["gen_t1"] - s["metadata"]["gen_t0"] for s in part]
        score = [s["metadata"]["opd_score_t1"] - s["metadata"]["opd_score_t0"] for s in part]
        lengths = [s["response_length"] for s in part]
        tokens = sum(len(s["tokens"]) for s in part)
        row[f"{task}_n"] = len(part)
        row[f"{task}_train_acc"] = float(np.mean([s["reward"] for s in part])) if part else float("nan")
        row[f"{task}_gen_p50_s"], row[f"{task}_gen_p90_s"] = _pct(gen, 50), _pct(gen, 90)
        row[f"{task}_len_p50"], row[f"{task}_len_p90"] = _pct(lengths, 50), _pct(lengths, 90)
        row[f"{task}_prompt_p50"] = _pct([len(s["tokens"]) - s["response_length"] for s in part], 50)
        row[f"{task}_score_p50_s"], row[f"{task}_score_p90_s"] = _pct(score, 50), _pct(score, 90)
        step_time = row.get("step_time_excl_eval") or rollout_time
        row[f"{task}_teacher_tok_per_s_per_gpu"] = tokens / step_time / teacher_gpus[task]
        row[f"{task}_teacher_step_mfu"] = (
            2 * args.active_params * tokens / (step_time * teacher_gpus[task] * args.peak_tflops * 1e12)
        )

    if row["train_time"]:
        row["trainer_tok_per_s_per_gpu"] = trained_tokens / row["train_time"] / args.trainer_gpus
        row["trainer_mfu"] = (
            6 * args.active_params * trained_tokens / (row["train_time"] * args.trainer_gpus * args.peak_tflops * 1e12)
        )
    completion = np.array(finish)
    return row, completion


def _steps_axis(*axes):
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def _line(ax, steps, values, label, color, marker="o", dashed=False):
    ax.plot(steps, values, label=label, color=color, marker=marker, linestyle="--" if dashed else "-")


def _plot_eval(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for task in TASKS:
        points = [(r["step"], r[f"eval_{task}"]) for r in rows if r.get(f"eval_{task}") is not None]
        if points:
            xs, ys = zip(*points, strict=True)
            _line(ax, xs, ys, f"student {task}", COLOR[task])
            ax.annotate(f"{ys[-1]:.1%}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)  # fmt: skip
        ax.axhline(TEACHER_ACC[task], color=COLOR[task], linestyle=":", linewidth=1.5,
                   label=f"teacher {task} ({TEACHER_ACC[task]:.1%})")  # fmt: skip
    ax.set(title="Held-out accuracy (avg@4, 128 prompts/task)", xlabel="update", ylabel="exact match", ylim=(0, 1.05))
    _steps_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_eval.png", dpi=200)
    plt.close(fig)


def _plot_step_time(rows, out):
    rows = [r for r in rows if r.get("step_time")]
    steps = [r["step"] for r in rows]
    for r in rows:
        r["_wait_excl_eval"] = (r.get("train_wait_time") or 0.0) - r.get("eval_s", 0.0)
    parts = [
        ("rollout + teacher scoring (wait)", "_wait_excl_eval", COLOR["caesar_cipher"]),
        ("old-policy forward", "log_probs_time", COLOR["simple_geometry"]),
        ("train forward + backward", "actor_train_time", COLOR["all"]),
        ("weight sync", "update_weights_time", "#eda100"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bottom = np.zeros(len(rows))
    for label, key, color in parts:
        values = np.array([r.get(key) or 0.0 for r in rows])
        ax.bar(steps, values, bottom=bottom, color=color, label=label, width=0.72, edgecolor=SURFACE, linewidth=1)
        bottom += values
    evals = np.array([r.get("eval_s", 0.0) for r in rows])
    other = np.clip(np.array([r["step_time"] for r in rows]) - evals - bottom, 0, None)
    ax.bar(steps, other, bottom=bottom, color="#c3c2b7", label="other", width=0.72, edgecolor=SURFACE, linewidth=1)
    bottom += other
    ax.bar(steps, evals, bottom=bottom, color=SURFACE, hatch="////", edgecolor="#8f8e88", linewidth=0.8,
           label="held-out eval (excluded from cost)", width=0.72)  # fmt: skip
    ax.set(title="Step time breakdown", xlabel="update", ylabel="seconds")
    _steps_axis(ax)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out / "fig_step_time.png", dpi=200)
    plt.close(fig)


def _plot_trainer(rows, out):
    rows = [r for r in rows if r.get("trainer_tok_per_s_per_gpu")]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    _line(axes[0], steps, [r["trainer_tok_per_s_per_gpu"] for r in rows], "trained tok/s/GPU", COLOR["all"])
    axes[0].set(title="Trainer throughput (train time only)", xlabel="update", ylabel="tokens / s / GPU")
    _line(axes[1], steps, [r["trainer_mfu"] for r in rows], "6·N·tokens", COLOR["all"])
    if any(r.get("actor_train_mfu") for r in rows):
        _line(axes[1], steps, [r.get("actor_train_mfu") for r in rows], "Miles counter, fwd+bwd only",
              COLOR["caesar_cipher"], marker="s")  # fmt: skip
    axes[1].set(title="Trainer MFU", xlabel="update", ylabel="MFU")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(fontsize=8)
    _steps_axis(*axes)
    fig.tight_layout()
    fig.savefig(out / "fig_trainer.png", dpi=200)
    plt.close(fig)


def _plot_rollout(rows, out):
    rows = [r for r in rows if r.get("trained_tokens")]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
    for task in TASKS:
        _line(axes[0], steps, [r[f"{task}_gen_p50_s"] for r in rows], f"{task} P50", COLOR[task])
        _line(axes[0], steps, [r[f"{task}_gen_p90_s"] for r in rows], f"{task} P90", COLOR[task], "^", True)
        _line(axes[1], steps, [r[f"{task}_len_p50"] for r in rows], f"{task} P50", COLOR[task])
        _line(axes[1], steps, [r[f"{task}_len_p90"] for r in rows], f"{task} P90", COLOR[task], "^", True)
    axes[0].set(title="Rollout end-to-end latency", xlabel="update", ylabel="seconds")
    axes[1].set(title="Response length", xlabel="update", ylabel="tokens")
    _line(axes[2], steps, [r["rollout_tok_per_s_per_gpu"] for r in rows], "generated tok/s/GPU", COLOR["all"])
    axes[2].set(title="Rollout throughput (nothing filtered)", xlabel="update", ylabel="tokens / s / GPU")
    for ax in axes[:2]:
        ax.legend(fontsize=7)
    _steps_axis(*axes)
    fig.tight_layout()
    fig.savefig(out / "fig_rollout.png", dpi=200)
    plt.close(fig)


def _plot_teacher(rows, out):
    rows = [r for r in rows if r.get("trained_tokens")]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
    for task in TASKS:
        seq = [r[f"{task}_prompt_p50"] + r[f"{task}_len_p50"] for r in rows]
        _line(axes[0], steps, seq, task, COLOR[task])
        _line(axes[1], steps, [r[f"{task}_score_p50_s"] for r in rows], f"{task} P50", COLOR[task])
        _line(axes[1], steps, [r[f"{task}_score_p90_s"] for r in rows], f"{task} P90", COLOR[task], "^", True)
        _line(axes[2], steps, [r[f"{task}_teacher_tok_per_s_per_gpu"] for r in rows], task, COLOR[task])
    axes[0].set(title="Teacher P50 sequence length", xlabel="update", ylabel="tokens")
    axes[1].set(title="Teacher scoring latency per rollout", xlabel="update", ylabel="seconds")
    axes[2].set(title="Teacher tokens scored / step time / GPU", xlabel="update", ylabel="tokens / s / GPU")
    for ax in axes:
        ax.legend(fontsize=7)
    _steps_axis(*axes)
    fig.tight_layout()
    fig.savefig(out / "fig_teacher.png", dpi=200)
    plt.close(fig)


def _plot_batch_completion(rows, completions, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    cmap = plt.get_cmap("Blues")
    steps = sorted(completions)
    for i, step in enumerate(steps):
        done = completions[step]
        frac = np.arange(1, len(done) + 1) / len(done)
        axes[0].step(done, frac, where="post", color=cmap(0.35 + 0.6 * i / max(len(steps) - 1, 1)), linewidth=1.2)
    axes[0].set(title="Batch completion per update (darker = later)", xlabel="seconds since rollout start",
                ylabel="fraction of batch scored")  # fmt: skip
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    timed = [r for r in rows if r.get("batch_last_done_s")]
    xs = [r["step"] for r in timed]
    _line(axes[1], xs, [r["batch_last_done_s"] for r in timed], "100% (last sample)", INK_2, "s")
    _line(axes[1], xs, [r["batch_p90_done_s"] for r in timed], "90% of batch", COLOR["caesar_cipher"], "^")
    _line(axes[1], xs, [r["batch_p50_done_s"] for r in timed], "50% of batch", COLOR["all"])
    axes[1].set(title="Time to finish 50% / 90% / 100% of the batch", xlabel="update", ylabel="seconds")
    axes[1].legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    _steps_axis(axes[1])
    fig.tight_layout()
    fig.savefig(out / "fig_batch_completion.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--trainer-gpus", type=int, required=True)
    parser.add_argument("--rollout-gpus", type=int, required=True)
    parser.add_argument("--teacher-gpus", nargs="+", required=True, help="TASK=N")
    parser.add_argument(
        "--total-gpus", type=int, default=None, help="Defaults to trainer (+ rollout if separate) + teachers"
    )
    parser.add_argument("--colocated", action="store_true", help="Trainer and rollout share GPUs")
    parser.add_argument("--peak-tflops", type=float, required=True, help="Dense BF16 per GPU: B200 2250, GB300 2500")
    parser.add_argument("--active-params", type=float, default=3e9)
    parser.add_argument("--gpu-hour-usd", type=float, default=1.73)
    args = parser.parse_args()
    teacher_gpus = {k: int(v) for k, v in (spec.split("=") for spec in args.teacher_gpus)}
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = _metrics_by_step(args.dump)
    samples = _samples_by_step(args.dump)
    eval_windows = _eval_windows(args.dump)
    rows, completions = [], {}
    for step in sorted(set(metrics) | set(samples)):
        row, completion = _step_row(step, metrics.get(step, {}), samples.get(step), args, teacher_gpus, eval_windows)
        rows.append(row)
        if completion is not None:
            completions[step] = completion

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    with (args.out / "steps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total_gpus = args.total_gpus or (
        args.trainer_gpus + (0 if args.colocated else args.rollout_gpus) + sum(teacher_gpus.values())
    )
    trained = [r for r in rows if r.get("trained_tokens") and r.get("step_time")]
    train_seconds = sum(r.get("step_time_excl_eval", r["step_time"]) for r in trained)
    trained_tokens = sum(r["trained_tokens"] for r in trained)
    gpu_hours = total_gpus * train_seconds / 3600

    def median(key):
        values = [r[key] for r in trained if r.get(key) is not None]
        return float(np.median(values)) if values else None

    summary = {
        "steps_trained": len(trained),
        "total_gpus": total_gpus,
        "training_wall_s_excl_eval_and_startup": train_seconds,
        "step_time_excl_eval_s": {
            "median": median("step_time_excl_eval"),
            "min": min((r["step_time_excl_eval"] for r in trained if "step_time_excl_eval" in r), default=None),
            "max": max((r["step_time_excl_eval"] for r in trained if "step_time_excl_eval" in r), default=None),
        },
        "eval_s_total": sum(t1 - t0 for t0, t1 in eval_windows),
        "batch_done_s_median": {"p50": median("batch_p50_done_s"), "p90": median("batch_p90_done_s"),
                                "last": median("batch_last_done_s")},  # fmt: skip
        "rollout_time_s_median": median("rollout_time"),
        "train_time_s_median": median("train_time"),
        "trainer_tok_per_s_per_gpu_median": median("trainer_tok_per_s_per_gpu"),
        "trainer_mfu_median": median("trainer_mfu"),
        "rollout_tok_per_s_per_gpu_median": median("rollout_tok_per_s_per_gpu"),
        "trained_tokens_M": trained_tokens / 1e6,
        "generated_tokens_M": sum(r["generated_tokens"] for r in trained) / 1e6,
        "dropped_rollouts": 0,
        "gpu_hours": gpu_hours,
        "usd_at_rate": gpu_hours * args.gpu_hour_usd,
        "usd_per_M_trained_tokens": gpu_hours * args.gpu_hour_usd / (trained_tokens / 1e6) if trained_tokens else None,
        "eval": {
            task: [(r["step"], r[f"eval_{task}"]) for r in rows if r.get(f"eval_{task}") is not None] for task in TASKS
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    _plot_eval(rows, args.out)
    _plot_step_time(rows, args.out)
    _plot_trainer(rows, args.out)
    _plot_rollout(rows, args.out)
    _plot_teacher(rows, args.out)
    _plot_batch_completion(rows, completions, args.out)


if __name__ == "__main__":
    main()
