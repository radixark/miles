"""Evaluate Terminal Bench 2.1 through a Harbor-compatible agent server.

The driver is resume-aware, streams one JSON object per task/trial to disk, and
writes a compact summary that ``ParallelCommandEvalFn`` can publish to W&B.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from tap import Tap


DEFAULT_TASK_LIST = Path(__file__).with_name("terminal_bench_2_1_tasks.txt")


class Args(Tap):
    agent_server_url: str
    """Harbor agent-server base URL reachable from this process."""

    base_url: str
    """OpenAI-compatible checkpoint endpoint reachable from the agent server."""

    model: str
    """LiteLLM model name, normally ``openai/<served-model-name>``."""

    output_jsonl: str
    summary_json: str
    task_list_file: str = str(DEFAULT_TASK_LIST)
    expected_tasks: int | None = 89
    n_trials: int = 2
    concurrency: int = 32
    agent_name: str = "terminus-2"
    max_seq_len: int = 262144
    max_tokens: int = 131072
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = 20
    per_task_timeout_sec: int = 12600
    api_key_env: str | None = None
    limit: int | None = None


def load_task_ids(path: Path, limit: int | None = None) -> list[str]:
    """Load a comment-tolerant, duplicate-free task list."""
    task_ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Task list contains duplicate IDs: {path}")
    return task_ids[:limit] if limit is not None else task_ids


def validate_args(args: Args, task_ids: list[str]) -> None:
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.max_seq_len <= 0 or args.max_tokens <= 0:
        raise ValueError("--max_seq_len and --max_tokens must be positive")
    if args.per_task_timeout_sec <= 0:
        raise ValueError("--per_task_timeout_sec must be positive")
    if args.expected_tasks is not None:
        expected = min(args.expected_tasks, args.limit) if args.limit is not None else args.expected_tasks
        if len(task_ids) != expected:
            raise ValueError(f"Expected {expected} Terminal Bench tasks, found {len(task_ids)}")


def resolve_api_key(args: Args) -> str:
    if args.api_key_env is None:
        return "dummy"
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env!r} is not set")
    return api_key


def build_payload(args: Args, instance_id: str, api_key: str) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.agent_name in {"terminus", "terminus-1", "terminus-2"}:
        sampling_params["api_key"] = api_key
    return {
        "base_url": args.base_url,
        "model": args.model,
        "instance_id": instance_id,
        "agent_name": args.agent_name,
        "max_seq_len": args.max_seq_len,
        "api_key": api_key,
        "sampling_params": sampling_params,
    }


def _decode_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text[:4000]}
    data = dict(payload) if isinstance(payload, Mapping) else {"raw": str(payload)[:4000]}
    data.setdefault("status_code", response.status_code)
    if response.status_code >= 400:
        data.setdefault("error", f"HTTP {response.status_code}")
    return data


async def run_one(
    client: httpx.AsyncClient,
    args: Args,
    api_key: str,
    instance_id: str,
    trial_idx: int,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        response = await client.post(
            f"{args.agent_server_url.rstrip('/')}/run",
            json=build_payload(args, instance_id, api_key),
            timeout=httpx.Timeout(float(args.per_task_timeout_sec), connect=30.0),
        )
        data = _decode_response(response)
    except httpx.TimeoutException:
        data = {"error": "timeout", "status_code": 0}
    except httpx.HTTPError as exc:
        data = {"error": f"{type(exc).__name__}: {exc}", "status_code": 0}
    data["instance_id"] = instance_id
    data["trial_idx"] = trial_idx
    data["elapsed_sec"] = round(time.monotonic() - started, 3)
    return data


def read_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("instance_id"), str):
                continue
            try:
                trial_idx = int(record.get("trial_idx", 0))
            except (TypeError, ValueError):
                continue
            records[(record["instance_id"], trial_idx)] = record
    return records


async def run_eval(
    args: Args,
    task_ids: list[str],
    output_path: Path,
    api_key: str,
) -> None:
    records = read_records(output_path)
    todo = [(instance_id, trial_idx) for instance_id in task_ids for trial_idx in range(args.n_trials) if (instance_id, trial_idx) not in records]
    print(
        f"Terminal Bench 2.1: {len(task_ids)} tasks x {args.n_trials} trials = {len(task_ids) * args.n_trials}; completed={len(records)}; todo={len(todo)}; concurrency={args.concurrency}",
        flush=True,
    )
    if not todo:
        return

    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)
    output_lock = asyncio.Lock()
    progress = {"completed": 0, "passes": 0, "started": time.monotonic()}

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 2)) as client:

        async def worker(worker_id: int) -> None:
            while True:
                try:
                    instance_id, trial_idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await run_one(client, args, api_key, instance_id, trial_idx)
                finally:
                    queue.task_done()
                reward = numeric_reward(record.get("reward"))
                async with output_lock:
                    with output_path.open("a") as handle:
                        handle.write(json.dumps(record, default=str) + "\n")
                    progress["completed"] += 1
                    progress["passes"] += int(reward == 1.0)
                    elapsed = max(time.monotonic() - progress["started"], 1e-6)
                    rate = progress["completed"] / elapsed
                    remaining = len(todo) - progress["completed"]
                    eta_minutes = remaining / rate / 60 if rate > 0 else 0.0
                    print(
                        f"[w{worker_id}] {progress['completed']}/{len(todo)} id={instance_id} trial={trial_idx} reward={reward} error={record.get('error', '-')} eta={eta_minutes:.1f}min",
                        flush=True,
                    )

        await asyncio.gather(*(worker(worker_id) for worker_id in range(args.concurrency)))


def numeric_reward(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    reward = float(value)
    return reward if math.isfinite(reward) else None


def _status_code(record: Mapping[str, Any] | None) -> int:
    if record is None:
        return 0
    try:
        return int(record.get("status_code", 0))
    except (TypeError, ValueError):
        return 0


def summarize(
    output_path: Path,
    task_ids: list[str],
    n_trials: int,
) -> dict[str, Any]:
    records = read_records(output_path)
    expected_pairs = [(instance_id, trial_idx) for instance_id in task_ids for trial_idx in range(n_trials)]
    expected_keys = set(expected_pairs)
    rewards: list[float | None] = []
    per_task: dict[str, dict[str, int | float]] = {}
    total_passes = 0
    total_graded = 0
    total_completed = 0
    request_successes = 0
    timeouts = 0
    sequence_length_exceeded = 0
    time_limit_exceeded = 0

    for instance_id in task_ids:
        task_passes = 0
        task_graded = 0
        task_completed = 0
        task_errors = 0
        for trial_idx in range(n_trials):
            record = records.get((instance_id, trial_idx))
            reward = numeric_reward(record.get("reward")) if record is not None else None
            rewards.append(reward)
            task_completed += int(record is not None)
            task_graded += int(reward is not None)
            task_passes += int(reward == 1.0)
            task_errors += int(reward is None)
            status_code = _status_code(record)
            request_successes += int(200 <= status_code < 300)
            error_text = "" if record is None else f"{record.get('error', '')} {record.get('exit_status', '')}".lower()
            timeouts += int("timeout" in error_text)
            sequence_length_exceeded += int("sequencelength" in error_text and "exceeded" in error_text)
            time_limit_exceeded += int("timelimitexceeded" in error_text)
        total_passes += task_passes
        total_graded += task_graded
        total_completed += task_completed
        per_task[instance_id] = {
            "trials_total": n_trials,
            "completed": task_completed,
            "graded": task_graded,
            "passes": task_passes,
            "failures": task_graded - task_passes,
            "errors": task_errors,
            "accuracy": task_passes / n_trials,
        }

    trials_total = len(expected_pairs)
    problems_any_correct = sum(int(task["passes"] > 0) for task in per_task.values())
    metrics: dict[str, int | float] = {
        "tasks_total": len(task_ids),
        "trials_per_task": n_trials,
        "trials_total": trials_total,
        "completed": total_completed,
        "graded": total_graded,
        "passes": total_passes,
        "failures": total_graded - total_passes,
        "errors": trials_total - total_graded,
        "missing": trials_total - total_completed,
        "accuracy": total_passes / trials_total if trials_total else 0.0,
        "request_success_rate": request_successes / trials_total if trials_total else 0.0,
        "timeouts": timeouts,
        "sequence_length_exceeded": sequence_length_exceeded,
        "time_limit_exceeded": time_limit_exceeded,
        "problems_any_correct": problems_any_correct,
        "problem_any_correct_rate": problems_any_correct / len(task_ids) if task_ids else 0.0,
        "unexpected_records": len(set(records) - expected_keys),
    }
    return {"metrics": metrics, "rewards": rewards, "per_task": per_task}


async def main_async(args: Args) -> None:
    task_ids = load_task_ids(Path(args.task_list_file), args.limit)
    validate_args(args, task_ids)
    api_key = resolve_api_key(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    await run_eval(args, task_ids, output_path, api_key)
    summary = summarize(output_path, task_ids, args.n_trials)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary["metrics"], sort_keys=True), flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)


def main() -> None:
    asyncio.run(main_async(Args().parse_args()))


if __name__ == "__main__":
    main()
