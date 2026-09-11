"""Run N copies of e2e/e2e_client.py at once against one gateway, one tenant each.

Every copy is its own process with its own api_key (``tml-e2e-user-NN``), so the gateway
sees N independent Tinker users sending the same requests; this file only starts them
together, relays their output under a tag, summarizes their timings, and stops them if the
gateway dies (the SDK would retry forever).

    python e2e/run_clients.py --n-clients 47 --summary-json summary.json -- --base-model ... --dataset ...
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import urllib.request

CLIENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_client.py")
PHASES = ("fwd_bwd", "optim", "publish", "rollout")
HEALTH_INTERVAL_S = 30
HEALTH_FAILURES_TO_ABORT = 3
_STEP_LINE = re.compile(r"\bstep=(\d+) loss=")
_SECONDS = re.compile(r"\b(\w+)=([0-9.]+)s\b")
_NUMBER = re.compile(r"\b(acc|mean_len|max_len|prompt_len)=([0-9.]+)%?")


def base_url(client_args: list[str]) -> str:
    if "--base-url" in client_args:
        return client_args[client_args.index("--base-url") + 1]
    return "http://127.0.0.1:9646"


def gateway_alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/v1/healthz", timeout=10) as response:
            return response.status == 200
    except OSError:
        return False


async def watch_gateway(url: str, processes: list) -> None:
    """Stop every client once the gateway has been unreachable for a while."""
    failures = 0
    while True:
        await asyncio.sleep(HEALTH_INTERVAL_S)
        failures = 0 if await asyncio.to_thread(gateway_alive, url) else failures + 1
        if failures >= HEALTH_FAILURES_TO_ABORT:
            print(
                f"[summary] gateway {url} unreachable for {failures * HEALTH_INTERVAL_S}s; stopping the clients",
                flush=True,
            )
            for process in processes:
                if process.returncode is None:
                    process.terminate()
            return


async def run_client(index: int, client_args: list[str], processes: list) -> tuple[str, int, list[str]]:
    tag = f"user-{index:02d}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        CLIENT,
        "--api-key",
        f"tml-e2e-{tag}",
        *client_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    processes.append(process)
    lines = []
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        print(f"[{tag}] {line}", flush=True)
        lines.append(line)
    return tag, await process.wait(), lines


def step_records(lines: list[str]) -> list[dict]:
    """One record per ``step=N loss=...`` line: the phase seconds and the sampled lengths."""
    records = []
    for line in lines:
        match = _STEP_LINE.search(line)
        if match is None:
            continue
        record = {"step": int(match.group(1))}
        record.update({key: float(value) for key, value in _SECONDS.findall(line) if key in PHASES})
        record.update({key: float(value) for key, value in _NUMBER.findall(line)})
        if all(phase in record for phase in PHASES):
            records.append(record)
    return records


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    quantile = lambda p: ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]  # noqa: E731
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize(results: list[tuple[str, int, list[str]]], elapsed_s: float) -> dict:
    """Across every client-step: the four phases, their share of a step, the step total, the lengths."""
    records = [record for _, _, lines in results for record in step_records(lines)]
    summary = {
        "n_clients": len(results),
        "passed": sum(code == 0 for _, code, _ in results),
        "failed": [tag for tag, code, _ in results if code != 0],
        "elapsed_s": elapsed_s,
        "client_steps": len(records),
        "phases": {},
        "per_step": {},
    }
    if not records:
        return summary
    totals = [sum(record[phase] for phase in PHASES) for record in records]
    total_mean = statistics.fmean(totals)
    for phase in PHASES:
        values = [record[phase] for record in records]
        summary["phases"][phase] = {**stats(values), "share": statistics.fmean(values) / total_mean}
    summary["step_total"] = stats(totals)
    for step in sorted({record["step"] for record in records}):
        rows = [record for record in records if record["step"] == step]
        summary["per_step"][str(step)] = {"clients": len(rows)}
        for key, reduce in (
            ("acc", statistics.fmean),
            ("mean_len", statistics.fmean),
            ("max_len", max),
            ("prompt_len", statistics.fmean),
        ):
            values = [row[key] for row in rows if key in row]
            if values:
                summary["per_step"][str(step)][key] = reduce(values)
    return summary


def print_summary(summary: dict) -> None:
    for phase, values in summary["phases"].items():
        print(
            f"[summary] {phase:8s} n={values['n']:4d} mean={values['mean']:7.1f}s p50={values['p50']:7.1f}s "
            f"p90={values['p90']:7.1f}s max={values['max']:7.1f}s share={values['share']:5.1%}",
            flush=True,
        )
    if "step_total" in summary:
        total = summary["step_total"]
        print(
            f"[summary] step total  mean={total['mean']:7.1f}s p50={total['p50']:7.1f}s "
            f"p90={total['p90']:7.1f}s max={total['max']:7.1f}s",
            flush=True,
        )


async def main(args) -> int:
    started = time.time()
    print(f"[summary] {args.n_clients} clients x e2e_client.py {' '.join(args.client_args)}", flush=True)
    processes: list = []
    watchdog = asyncio.create_task(watch_gateway(base_url(args.client_args), processes))
    results = await asyncio.gather(
        *(run_client(index, args.client_args, processes) for index in range(args.n_clients))
    )
    watchdog.cancel()
    summary = summarize(results, time.time() - started)
    print_summary(summary)
    if args.summary_json:
        with open(args.summary_json, "w") as handle:
            json.dump(summary, handle, indent=2)
    if summary["failed"]:
        print(
            f"[summary] FAIL: {len(summary['failed'])}/{args.n_clients} clients failed "
            f"({summary['elapsed_s']:.0f}s): {' '.join(summary['failed'])}",
            flush=True,
        )
        return 1
    print(f"[summary] PASS: {args.n_clients}/{args.n_clients} clients ({summary['elapsed_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, required=True)
    parser.add_argument("--summary-json", help="where to write the timing summary the run report reads")
    parser.add_argument("client_args", nargs=argparse.REMAINDER, help="arguments passed to every e2e_client.py")
    parsed = parser.parse_args()
    if parsed.client_args[:1] == ["--"]:
        parsed.client_args = parsed.client_args[1:]
    raise SystemExit(asyncio.run(main(parsed)))
