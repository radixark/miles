"""Watch a running TMax Beaker experiment and print one line per event.

    python examples/tmax/tools/watch_run.py <experiment> [--every-step 10] [--slow-sync 20] [--until-steps N]

Events:

* `STEP n ...`: a trained step (every --every-step-th), with loss, grad norm, clip fraction and train/rollout log-prob
  gap;
* `SYNC ...`: a weight sync that failed, or took longer than --slow-sync seconds (all of them at 0);
* `FAIL: ...`: a failure signal in the head log (a failed cell call, Ray task errors, NCCL errors, OOM, a trainer
  traceback, engines being killed);
* `REPLICA EXITED ...` for every replica job that exited, and `HEAD EXITED` before the watcher stops.

Every --interval seconds it reads `beaker experiment get` and the head job's log (replica 0) since its last successful
read. It remembers what it printed, and when it last read the log, in --state-dir, so a restarted watcher neither
repeats itself nor skips what was logged while it was down (up to --max-gap), and appends the raw log there
(`head.log`). It stops when the head job exits or, with --until-steps N, once N steps trained and a weight sync
followed each. Each stdout line is one event, so it suits a monitor that turns lines into notifications.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP = re.compile(r"rank00000\] model\.py:\d+ - step (\d+): \{'train/loss'")
_STEP_METRICS = re.compile(r"'train/(loss|grad_norm|pg_clipfrac|train_rollout_logprob_abs_diff)': ([-0-9.e]+)")
_SYNC = re.compile(
    r"^(\S+) .*trainer_controller_actor\] cell\.py:\d+ - ft op=execute phase=(?:end|fail) .*fn=update_weights (.*)$"
)
_FAIL = re.compile(
    r"phase=fail|ok=false|_mark_as_errored|Killing server|RayTaskError|ActorDiedError|out of memory|OutOfMemoryError"
    r"|Watchdog caught|NCCL error|nccl(?:Internal|Remote|System)Error|\((?:MegatronTrainRayActor|TrainerController)[^)]*\) Traceback"
)
_LOG_TIMESTAMP = re.compile(r"^\S+ |\[[0-9-]+ [0-9:.]+ ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="Beaker experiment id or name")
    parser.add_argument("--state-dir", default="", help="default: /tmp/tmax-watch-<experiment>")
    parser.add_argument("--every-step", type=int, default=1, help="report every N-th trained step")
    parser.add_argument("--slow-sync", type=float, default=0, help="report only syncs that fail or exceed this (s)")
    parser.add_argument("--until-steps", type=int, default=0, help="stop once this many steps trained and synced")
    parser.add_argument("--interval", type=float, default=90)
    parser.add_argument("--max-gap", type=float, default=3 * 3600, help="read at most this far back (s)")
    args = parser.parse_args()

    state = Path(args.state_dir or f"/tmp/tmax-watch-{re.sub(r'[^A-Za-z0-9_-]', '_', args.experiment)}")
    state.mkdir(parents=True, exist_ok=True)
    seen = _Seen(state / "seen.txt")
    steps = _Seen(state / "steps.txt")
    syncs = _Seen(state / "syncs.txt")
    last_read = state / "last_read.txt"
    while True:
        jobs = _jobs(args.experiment)
        heads = [job for job in jobs or [] if job["rank"] == 0]
        if not heads:  # beaker unreachable, or no job scheduled yet
            time.sleep(args.interval)
            continue
        for job in jobs:
            if job["exited"]:
                seen.emit(f"REPLICA EXITED rank={job['rank']} job={job['id']} at={job['exited']} code={job['code']}")
        head = heads[-1]  # a retry adds a new job per replica
        # Read from the last successful read, with a minute of overlap (`seen` drops the repeats): a slow or failed
        # read, or a watcher restarted after a while, leaves no gap.
        started = time.time()
        previous = float(last_read.read_text()) if last_read.exists() else started - args.interval - 120
        log = _head_log(head["id"], f"{int(min(started - previous, args.max_gap)) + 60}s")
        if log is not None:
            last_read.write_text(str(started))
            with open(state / "head.log", "a") as f:
                f.write(log)
            _report(log, args, seen, steps, syncs)
        if head["exited"]:
            print(f"HEAD EXITED job={head['id']} code={head['code']}", flush=True)
            return
        if args.until_steps and len(steps) >= args.until_steps and len(syncs) > args.until_steps:
            print(f"DONE: {len(steps)} steps trained, {len(syncs)} weight syncs ok", flush=True)
            return
        time.sleep(args.interval)


def _report(log: str, args, seen: "_Seen", steps: "_Seen", syncs: "_Seen") -> None:
    failures = []
    for line in log.splitlines():
        if match := _STEP.search(line):
            step = int(match.group(1))
            if steps.add(str(step)) and step % args.every_step == 0:
                metrics = " ".join(f"{key}={value}" for key, value in _STEP_METRICS.findall(line))
                seen.emit(f"STEP {step} {line[:24]} {metrics}")
        elif match := _SYNC.search(line):
            timestamp, outcome = match.groups()
            elapsed = float(re.search(r"elapsed_s=([0-9.]+)", outcome).group(1)) if "elapsed_s=" in outcome else 0.0
            ok = "ok=true" in outcome
            if ok:
                syncs.add(timestamp)
            if not ok or elapsed > args.slow_sync:
                seen.emit(f"SYNC {timestamp} {outcome}")
        elif _FAIL.search(line):
            failures.append(_LOG_TIMESTAMP.sub("", line, count=2)[:220])
    for failure in list(dict.fromkeys(failures))[:5]:
        seen.emit(f"FAIL: {failure}")


def _jobs(experiment: str) -> list[dict] | None:
    result = subprocess.run(
        ["beaker", "experiment", "get", experiment, "--format", "json"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"beaker experiment get failed: {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    data = json.loads(result.stdout)
    data = data[0] if isinstance(data, list) else data
    return [
        {
            "id": job["id"],
            "rank": (job.get("execution") or {}).get("replicaRank") or 0,
            "exited": (job.get("status") or {}).get("exited") or "",
            "code": (job.get("status") or {}).get("exitCode", ""),
        }
        for job in data.get("jobs") or []
    ]


def _head_log(job_id: str, since: str) -> str | None:
    result = subprocess.run(["beaker", "job", "logs", job_id, "--since", since], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"beaker job logs failed: {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return _ANSI.sub("", result.stdout)


class _Seen:
    """A set of lines persisted to a file."""

    def __init__(self, path: Path):
        self._path = path
        self._lines = set(path.read_text().splitlines()) if path.exists() else set()

    def __len__(self) -> int:
        return len(self._lines)

    def add(self, line: str) -> bool:
        if line in self._lines:
            return False
        self._lines.add(line)
        with open(self._path, "a") as f:
            f.write(line + "\n")
        return True

    def emit(self, line: str) -> None:
        if self.add(line):
            print(line, flush=True)


if __name__ == "__main__":
    main()
