"""Evaluate Kimi Code on AgenticRL-Bench tasks with a plain API policy.

The runner is standalone: no training process, session server, or agent server is
required. Harbor runs each task and its verifier in the selected sandbox backend;
Kimi Code calls an OpenAI-compatible API from inside the task environment.

Example:
    export POLICY_BASE_URL=https://api.example.com/v1
    export KIMI_MODEL_API_KEY=...
    python examples/experimental/eval/agenticrl_bench_via_api/eval_agenticrl_via_api.py \
        --tasks-dir /path/to/agenticrl-bench/tasks \
        --task algorithm-partial_rollout-branch-reset-fidelity \
        --attempts 1 --max-concurrent 1
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

EXAMPLE_DIR = Path(__file__).resolve().parent
ENVIRONMENT_HOSTS = (
    "pypi.org",
    "files.pythonhosted.org",
    "astral.sh",
    "releases.astral.sh",
    "code.kimi.com",
    "cdn.kimi.com",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "raw.githubusercontent.com",
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "registry.npmjs.org",
    "nodejs.org",
    "deb.nodesource.com",
    "storage.googleapis.com",
    "*.archive.ubuntu.com",
    "ports.ubuntu.com",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[], help="Task directory name; repeat to select tasks")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--attempt-start", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs/agenticrl-kimi-code"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("agenticrl-kimi-code-results.jsonl"))
    parser.add_argument("--model", default="openai/kimi-k3.1")
    parser.add_argument("--base-url", default=os.environ.get("POLICY_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="KIMI_MODEL_API_KEY")
    parser.add_argument("--environment", default="agentenv")
    parser.add_argument("--environment-kwarg", action="append", default=[])
    parser.add_argument("--max-lifetime-secs", type=int, default=39600)
    parser.add_argument("--max-context-size", type=int, default=1048576)
    parser.add_argument("--thinking-effort", default="max")
    parser.add_argument("--system-dns", action="store_true", help="Use the OS DNS resolver in the Harbor client")
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.attempt_start < 1 or args.max_concurrent < 1:
        parser.error("--attempts, --attempt-start, and --max-concurrent must be positive")
    if not args.base_url or not urlsplit(args.base_url).hostname:
        parser.error("set --base-url or POLICY_BASE_URL to an OpenAI-compatible API root")
    return args


def _select_tasks(tasks_dir: Path, selected: list[str]) -> list[Path]:
    available = sorted(path for path in tasks_dir.iterdir() if (path / "task.toml").is_file())
    by_name = {path.name: path for path in available}
    missing = sorted(set(selected) - set(by_name))
    if missing:
        raise ValueError(f"tasks not found under {tasks_dir}: {', '.join(missing)}")
    tasks = [by_name[name] for name in selected] if selected else available
    if not tasks:
        raise ValueError(f"no tasks found under {tasks_dir}")
    return tasks


def _format_kwarg(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _environment_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_lifetime_secs": args.max_lifetime_secs}
    for item in args.environment_kwarg:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--environment-kwarg must be KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        kwargs[key] = value
    return kwargs


def _build_command(args: argparse.Namespace, task: Path, attempt: int, output_dir: Path) -> list[str]:
    command = [
        "harbor",
        "run",
        "--yes",
        "--path",
        str(task),
        "--agent",
        "kimi_code_agent:KimiCodeAgent",
        "--model",
        args.model,
        "--agent-env",
        "OPENAI_API_KEY=${OPENAI_API_KEY}",
        "--env",
        args.environment,
        "--job-name",
        f"{task.name}__kimi-code__a{attempt}",
        "--jobs-dir",
        str(output_dir),
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--allow-agent-host",
        urlsplit(args.base_url).hostname,
    ]
    for key, value in _environment_kwargs(args).items():
        command.extend(("--environment-kwarg", f"{key}={_format_kwarg(value)}"))
    for host in ENVIRONMENT_HOSTS:
        command.extend(("--allow-environment-host", host))
    return command


def _child_env(args: argparse.Namespace, api_key: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": api_key,
            "KIMI_MODEL_BASE_URL": args.base_url,
            "KIMI_MODEL_MAX_CONTEXT_SIZE": str(args.max_context_size),
            "KIMI_MODEL_THINKING_EFFORT": args.thinking_effort,
            "KIMI_EVAL_SYSTEM_DNS": "1" if args.system_dns else "0",
            "PYTHONPATH": os.pathsep.join([str(EXAMPLE_DIR), env.get("PYTHONPATH", "")]),
        }
    )
    if not env.get("E2B_API_KEY") and env.get("E2B_API_KEY_FILE"):
        env["E2B_API_KEY"] = Path(env["E2B_API_KEY_FILE"]).expanduser().read_text().strip()
    return env


def _read_trial_result(output_dir: Path) -> dict[str, Any]:
    trial_results = [path for path in output_dir.rglob("result.json") if path.is_file()]
    for path in trial_results:
        result = json.loads(path.read_text())
        if "verifier_result" not in result and "exception_info" not in result:
            continue
        exception = result.get("exception_info")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        return {
            "reward": rewards.get("reward", rewards.get("gate")),
            "exit_status": exception.get("exception_type") if exception else "Submitted",
            "trial_dir": str(path.parent),
        }
    path = output_dir / "result.json"
    if not path.is_file():
        return {"reward": None, "exit_status": "MissingResult", "trial_dir": ""}
    result = json.loads(path.read_text())
    evals = (result.get("stats") or {}).get("evals") or {}
    metrics = next((value.get("metrics") or [] for value in evals.values()), [])
    reward = metrics[0].get("reward", metrics[0].get("gate")) if metrics else None
    stats = result.get("stats") or {}
    exit_status = "Submitted" if stats.get("n_completed_trials") and not stats.get("n_errored_trials") else "Error"
    return {"reward": reward, "exit_status": exit_status, "trial_dir": ""}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_attempt(args: argparse.Namespace, api_key: str, task: Path, attempt: int) -> dict[str, Any]:
    output_dir = args.jobs_dir / task.name / f"attempt-{attempt:02d}"
    started_at = _utc_now()
    started = asyncio.get_running_loop().time()
    row: dict[str, Any] = {"task": task.name, "attempt": attempt, "started_at": started_at}
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        command = _build_command(args, task, attempt, output_dir)
        config = {
            "task": str(task),
            "attempt": attempt,
            "model": args.model,
            "base_url": args.base_url,
            "environment": args.environment,
            "command": command,
        }
        (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        with (output_dir / "console.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=_child_env(args, api_key),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            returncode = await process.wait()
        row.update(_read_trial_result(output_dir))
        row["process_exit_code"] = returncode
    except Exception as exc:
        row.update({"reward": None, "exit_status": "RunnerError", "error": f"{type(exc).__name__}: {exc}"})
    row["finished_at"] = _utc_now()
    row["wall_seconds"] = round(asyncio.get_running_loop().time() - started, 3)
    row["job_dir"] = str(output_dir)
    return row


async def main() -> int:
    args = _parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} in the local environment; it will not be logged")
    tasks = _select_tasks(args.tasks_dir, args.task)
    attempts = [
        (task, attempt) for task in tasks for attempt in range(args.attempt_start, args.attempt_start + args.attempts)
    ]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def bounded(task: Path, attempt: int) -> dict[str, Any]:
        async with semaphore:
            return await _run_attempt(args, api_key, task, attempt)

    results: list[dict[str, Any]] = []
    async with asyncio.TaskGroup() as group:
        pending = [group.create_task(bounded(task, attempt)) for task, attempt in attempts]
        for future in asyncio.as_completed(pending):
            row = await future
            results.append(row)
            with args.output_jsonl.open("a") as output:
                output.write(json.dumps(row) + "\n")
            print(
                f"[{len(results)}/{len(attempts)}] {row['task']} attempt={row['attempt']} "
                f"reward={row.get('reward')} exit_status={row.get('exit_status')} "
                f"seconds={row['wall_seconds']}",
                flush=True,
            )
    passed = sum(row.get("reward") == 1.0 for row in results)
    runner_errors = sum(row.get("process_exit_code", 1) != 0 for row in results)
    print(f"Completed {len(results)} attempts: {passed} passed, {runner_errors} runner errors")
    return 1 if runner_errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
