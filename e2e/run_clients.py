"""Run N copies of e2e/e2e_client.py at once against one gateway, one tenant each.

Every copy is its own process with its own api_key (``tml-e2e-user-NN``), so the gateway
sees N independent Tinker users sending the same requests; this file only starts them
together, relays their output under a tag, and summarizes their timings.

    python e2e/run_clients.py --n-clients 47 -- --base-model ... --dataset ... --steps 3
"""

import argparse
import asyncio
import os
import re
import statistics
import sys
import time
import urllib.request

CLIENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_client.py")
PHASES = ("fwd_bwd", "optim", "publish", "rollout")
HEALTH_INTERVAL_S = 30
HEALTH_FAILURES_TO_ABORT = 3  # the SDK retries a dead gateway forever; the run must not
_STEP_LINE = re.compile(r"\bstep=(\d+)\b")
_PHASE_VALUE = re.compile(r"\b(\w+)=([0-9.]+)s\b")


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


def phase_seconds(lines: list[str]) -> dict[str, list[float]]:
    """Per phase, the seconds every ``step=N ... phase=Xs`` line reports."""
    seconds = {phase: [] for phase in PHASES}
    for line in lines:
        match = _STEP_LINE.search(line)
        if match is None or match.group(1) == "0":
            continue
        for phase, value in _PHASE_VALUE.findall(line):
            if phase in seconds:
                seconds[phase].append(float(value))
    return seconds


def summarize(results: list[tuple[str, int, list[str]]]) -> None:
    """Across every client-step: mean / p50 / p90 / max of each phase (the waits a tenant sees)."""
    seconds = {phase: [] for phase in PHASES}
    for _, _, lines in results:
        for phase, values in phase_seconds(lines).items():
            seconds[phase].extend(values)
    for phase, values in seconds.items():
        if not values:
            continue
        ordered = sorted(values)
        p50, p90 = ordered[len(ordered) // 2], ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        print(
            f"[summary] {phase:8s} n={len(values):4d} mean={statistics.fmean(values):7.1f}s "
            f"p50={p50:7.1f}s p90={p90:7.1f}s max={ordered[-1]:7.1f}s",
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
    summarize(results)
    failed = [tag for tag, code, _ in results if code != 0]
    elapsed = time.time() - started
    if failed:
        print(f"[summary] FAIL: {len(failed)}/{args.n_clients} clients failed ({elapsed:.0f}s): {' '.join(failed)}")
        return 1
    print(f"[summary] PASS: {args.n_clients}/{args.n_clients} clients ({elapsed:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, required=True)
    parser.add_argument("client_args", nargs=argparse.REMAINDER, help="arguments passed to every e2e_client.py")
    parsed = parser.parse_args()
    if parsed.client_args[:1] == ["--"]:
        parsed.client_args = parsed.client_args[1:]
    raise SystemExit(asyncio.run(main(parsed)))
