"""The run report auto_e2e_test.sh prints at the end: one box table with the setup (nodes,
task, sequence lengths), then the results (slots, SGLang knobs, outcome, the four phases a
client waits through with their share of a step, and the GPU peaks of each node).

Inputs: the run dir (serve.log, client-summary.json, gpu-<ip>.csv from e2e/gpu_sampler.sh)
and the harness knobs, passed as --knob key=value.
"""

import argparse
import csv
import json
import os
import re
import textwrap

WIDTHS = (31, 101)
SETUP_ROWS = 4  # the heavy rule goes after these


def knob_dict(pairs: list[str]) -> dict:
    return dict(pair.split("=", 1) for pair in pairs)


def serve_facts(path: str) -> dict:
    """What the gateway logged: the resolved slots and bounds, the adapter cap, the roles of the nodes."""
    facts = {"trainer_ips": set(), "engine_ips": set()}
    with open(path, errors="replace") as handle:
        for line in handle:
            if "multi-LoRA capacity:" in line and "slots" not in facts:
                facts["slots"] = int(re.search(r"capacity: (\d+) slots", line).group(1))
                facts["binding"] = re.search(r"bound by (.+?) [\(\[]", line).group(1)
                facts["bounds"] = re.search(r"\[(trainer memory=.*?)\] \(slot=", line).group(1)
            elif "adapter versions loaded" in line:
                facts["loaded_loras"] = int(re.search(r"at most (\d+) adapter", line).group(1))
            elif "MegatronTrainRayActor pid=" in line:
                facts["trainer_ips"].add(re.search(r"ip=([0-9.]+)", line).group(1))
            elif "Uvicorn running on http://" in line and "CommandActor" in line:
                facts["engine_ips"].add(re.search(r"http://([0-9.]+):", line).group(1))
    return facts


def dataset_label(path: str) -> str:
    """gsm8k/train.parquet rather than train.parquet; a distinctive file name stands alone."""
    name = os.path.basename(path)
    if name.split(".")[0] in {"train", "test", "validation", "data"}:
        return f"{os.path.basename(os.path.dirname(path))}/{name}"
    return name


def gpu_peaks(path: str) -> dict | None:
    """Peak memory (MiB and % of total) and peak SM utilization over every sample of every GPU."""
    used, total, util = [], None, []
    with open(path) as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            used.append(int(row[2].split()[0]))
            total = int(row[3].split()[0])
            util.append(int(row[4].split()[0]))
    if not used:
        return None
    return {"mem_peak": max(used), "mem_total": total, "util_peak": max(util), "samples": len(used)}


def gpu_rows(run_dir: str, facts: dict, knobs: dict) -> list[tuple[str, str]]:
    rows = []
    for name in sorted(os.listdir(run_dir)):
        if not (name.startswith("gpu-") and name.endswith(".csv")):
            continue
        ip = name[4:-4]
        peaks = gpu_peaks(os.path.join(run_dir, name))
        if peaks is None:
            continue
        if ip in facts["trainer_ips"] and ip not in facts["engine_ips"]:
            role, gpus = "trainer", int(knobs["TRAIN_GPUS"])
        elif ip in facts["engine_ips"] and ip not in facts["trainer_ips"]:
            role, gpus = "SGLang", int(knobs["ROLLOUT_GPUS"])
        else:
            role, gpus = f"node {ip}", 8
        rows.append(
            (
                f"{role} GPUs ({gpus})",
                f"peak memory {peaks['mem_peak']:,} MiB / {peaks['mem_total']:,} MiB = "
                f"{100 * peaks['mem_peak'] / peaks['mem_total']:.1f}%; peak SM utilization {peaks['util_peak']}%",
            )
        )
    return rows


def phase_row(label: str, values: dict) -> tuple[str, str]:
    return (
        f"{label} (time/step)",
        f"mean {values['mean']:.0f} / p50 {values['p50']:.0f} / p90 {values['p90']:.0f} / max {values['max']:.0f} s"
        + (f"; {values['share']:.1%} of a step" if "share" in values else "; 100%"),
    )


def build_rows(knobs: dict, facts: dict, summary: dict, gpu: list[tuple[str, str]]) -> list[tuple[str, str]]:
    engines = int(knobs["ROLLOUT_GPUS"]) // int(knobs["GPUS_PER_ENGINE"])
    steps = int(knobs["STEPS"])
    per_step = summary.get("per_step", {})
    lengths = [step["mean_len"] for step in per_step.values() if step.get("mean_len")]
    max_len = max((step["max_len"] for step in per_step.values() if step.get("max_len")), default=None)
    prompts = [step["prompt_len"] for step in per_step.values() if step.get("prompt_len")]
    setup = [
        (
            "nodes",
            f"{knobs['TRAIN_NODES']} trainer node(s) x {knobs['TRAIN_GPUS']} GPUs TP{knobs['TP']}/EP{knobs['EP']}, "
            f"{engines} x TP{knobs['GPUS_PER_ENGINE']} SGLang engines on {knobs['ROLLOUT_GPUS']} GPUs ({knobs['GPU_NAME']})",
        ),
        (
            "task",
            f"DAPO on {dataset_label(knobs['DATASET'])} ({os.path.basename(knobs['MODEL'])}, LoRA rank {knobs['LORA_RANK']}, "
            f"{knobs['PROMPTS_PER_STEP']} prompts x {knobs['SAMPLES_PER_PROMPT']} samples per step, lr {knobs['LR']}, {steps} steps)",
        ),
        (
            "input seq len",
            f"prompt up to {knobs['MAX_PROMPT_TOKENS']} tokens"
            + (f" (observed mean {sum(prompts) / len(prompts):.0f})" if prompts else "")
            + f"; context (prompt + output) {knobs['CONTEXT_LEN']} tokens",
        ),
        (
            "output seq len",
            f"up to {knobs['MAX_NEW_TOKENS']} tokens, clamped to the context left after the prompt"
            + (
                f"; observed mean {sum(lengths) / len(lengths):.0f} tokens "
                f"({' / '.join(f'{v:.0f}' for v in lengths)} per step)" + (f", max {max_len:.0f}" if max_len else "")
                if lengths
                else ""
            ),
        ),
    ]
    outcome = (
        f"{summary['passed']}/{summary['n_clients']} clients x {steps} steps passed; {summary['n_clients']} clients running "
        f"concurrently, {summary['elapsed_s'] / steps:.0f} s per step on average ({summary['elapsed_s']:.0f} s total); "
        f"gateway ready in {knobs.get('READY_S', '?')} s"
    )
    sglang = (
        f"mem-fraction {knobs['SGLANG_MEM_FRACTION']}, max-running {knobs['SGLANG_MAX_RUNNING_REQUESTS']}, "
        f"cuda-graph-bs {knobs['SGLANG_CUDA_GRAPH_MAX_BS']}, {knobs['SGLANG_MOE_RUNNER']} MoE, "
        f"max-loaded-loras {facts.get('loaded_loras', knobs.get('SGLANG_MAX_LOADED_LORAS') or '?')}"
        + (" (derived from the slot count)" if "loaded_loras" in facts else "")
    )
    results = [
        ("slots", str(facts.get("slots", knobs.get("N_ADAPTERS")))),
        ("SGLang", sglang),
        ("Outcome (time/step, whole run)", outcome),
    ]
    phases = summary.get("phases", {})
    for phase, label in (
        ("fwd_bwd", "fwd/bwd wait"),
        ("optim", "optim wait"),
        ("publish", "publish"),
        ("rollout", "rollout"),
    ):
        if phase in phases:
            results.append(phase_row(label, phases[phase]))
    if "step_total" in summary:
        results.append(phase_row("single client, one step total", summary["step_total"]))
    return setup + results + gpu


def render(rows: list[tuple[str, str]]) -> str:
    w1, w2 = WIDTHS

    def cell(text: str, width: int) -> str:
        return " " + text.ljust(width - 1)

    def row(key: str, value: str) -> str:
        keys = textwrap.wrap(key, w1 - 2) or [""]
        values = textwrap.wrap(value, w2 - 2) or [""]
        height = max(len(keys), len(values))
        keys += [""] * (height - len(keys))
        values += [""] * (height - len(values))
        return "\n".join(f"│{cell(k, w1)}│{cell(v, w2)}│" for k, v in zip(keys, values, strict=True))

    thin = f"├{'─' * w1}┼{'─' * w2}┤"
    heavy = f"┝{'━' * w1}┿{'━' * w2}┥"
    lines = [f"┌{'─' * w1}┬{'─' * w2}┐", f"│{'Item'.center(w1)}│{'Result'.center(w2)}│", thin]
    for index, (key, value) in enumerate(rows):
        lines.append(row(key, value))
        if index == SETUP_ROWS - 1:
            lines += [heavy, heavy]
        elif index < len(rows) - 1:
            lines.append(thin)
    lines.append(f"└{'─' * w1}┴{'─' * w2}┘")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--knob", action="append", default=[], help="key=value, repeatable")
    args = parser.parse_args()
    knobs = knob_dict(args.knob)
    facts = serve_facts(os.path.join(args.run_dir, "serve.log"))
    summary_path = os.path.join(args.run_dir, "client-summary.json")
    summary = (
        json.load(open(summary_path))
        if os.path.exists(summary_path)
        else {"n_clients": 0, "passed": 0, "elapsed_s": 0.0}
    )
    table = render(build_rows(knobs, facts, summary, gpu_rows(args.run_dir, facts, knobs)))
    notes = (
        f"slots: {facts.get('slots', '?')}, bound by {facts.get('binding', '?')} [{facts.get('bounds', '?')}]. "
        "time/step rows: what one LoRA observes in a single step, over every client-step, queueing included; "
        "shares divide each phase's mean by the step total's mean. GPU rows: peaks of nvidia-smi sampled every 15 s "
        "on every GPU of the node while the clients ran."
    )
    output = table + "\n" + "\n".join(textwrap.wrap(notes, 134))
    print(output, flush=True)
    with open(os.path.join(args.run_dir, "report.txt"), "w") as handle:
        handle.write(output + "\n")


if __name__ == "__main__":
    main()
