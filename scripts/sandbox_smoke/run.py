"""Golden smoke of one (connector, backend, agent, benchmark) combination on the real sandbox API.

PASS iff the verifier returns reward 1.0. What a run proves, the axes,
credentials, and how to choose what to run: README.md next to this file.
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

# The whole smoke should finish in minutes; a hung platform call must not eat
# hours. Overridable (AGENT_TRIAL_TIMEOUT) for slow first-time template builds.
_DEFAULT_TRIAL_TIMEOUT_S = "1200"

# The connector-neutral name for "execute the task's own reference solution".
GOLDEN = "golden"


@dataclass(frozen=True)
class Benchmark:
    tasks_dir: Callable[[], Path]  # resolved lazily: may fetch on first use
    smoke_task: str  # the instance a smoke run uses when --task is not given


_TB2_REPO = "https://github.com/laude-institute/terminal-bench-2.git"


def _tb2_tasks_dir() -> Path:
    """A Terminal-Bench-2 checkout: TB2_TASKS_DIR if set, else a cached shallow clone.

    TB2 task directories are native Harbor tasks (instruction.md, task.toml
    with a prebuilt official docker_image, tests/, solution/), so a checkout
    is directly usable as HARBOR_TASKS_DIR -- no preparation step.
    """
    env = os.environ.get("TB2_TASKS_DIR", "").strip()
    if env:
        return Path(env)
    cache = Path.home() / ".cache" / "miles-sandbox-smoke" / "terminal-bench-2"
    if not (cache / "fix-git").is_dir():
        print(f"sandbox-smoke: cloning {_TB2_REPO} -> {cache}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # clone into a scratch dir and rename into place, so an interrupted
        # clone leaves nothing at the final path to block the next run
        scratch = cache.with_name(cache.name + ".partial")
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", _TB2_REPO, str(scratch)], check=True)
        scratch.rename(cache)
    return cache


BENCHMARKS: dict[str, Benchmark] = {
    # fix-git: small, easy, the task our golden runs have always used.
    "tb2": Benchmark(tasks_dir=_tb2_tasks_dir, smoke_task="fix-git"),
}


async def _run_harbor(tasks_dir: Path, task: str, *, backend: str, agent: str, base_url: str) -> dict[str, Any]:
    """One trial through examples/experimental/harbor/harbor_agent_function."""
    sys.path.insert(0, str(REPO))  # for miles.rollout.agentic; no PYTHONPATH needed
    sys.path.insert(0, str(REPO / "examples" / "experimental" / "harbor"))
    try:
        import harbor  # noqa: F401 -- probed eagerly: harbor_agent_function defers its harbor imports
        import harbor_agent_function as haf
    except ImportError as e:
        raise SystemExit(
            f"{e}\nharbor is not importable in this environment; the install line is in {Path(__file__).parent / 'README.md'}"
        ) from e

    if agent == GOLDEN:
        agent = "oracle"  # harbor's solution-executing agent
    os.environ["HARBOR_TASKS_DIR"] = str(tasks_dir)
    os.environ["HARBOR_ENV_TYPE"] = backend
    os.environ.setdefault("AGENT_TRIAL_TIMEOUT", _DEFAULT_TRIAL_TIMEOUT_S)
    return await haf.run(
        base_url=base_url,
        prompt=[],
        request_kwargs={},
        metadata={"instance_id": task, "agent_name": agent},
    )


def _run_openenv(tasks_dir: Path, task: str, *, backend: str, agent: str, base_url: str) -> Any:
    raise NotImplementedError(
        "the openenv connector is not wired into this driver yet; its golden episode "
        "lives in examples/experimental/openenv/scan_golden.py until then"
    )


# connector name -> (tasks_dir, task, backend=, agent=, base_url=) -> result dict
CONNECTORS: dict[str, Callable[..., Any]] = {
    "harbor": _run_harbor,
    "openenv": _run_openenv,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connector", required=True, choices=sorted(CONNECTORS))
    parser.add_argument(
        "--backend", required=True, help="sandbox platform, passed through to the connector (e2b, daytona, modal, ...)"
    )
    parser.add_argument(
        "--agent",
        default=GOLDEN,
        help=f"{GOLDEN!r} (default, no model needed) or a real harness name (needs --base-url)",
    )
    parser.add_argument("--benchmark", default="tb2", choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--base-url", default="", help="OpenAI-compatible endpoint; required for any agent other than the golden one"
    )
    parser.add_argument("--task", default="", help="override the benchmark's preset smoke instance")
    parser.add_argument("--tasks-dir", type=Path, default=None, help="override the benchmark's task directory")
    args = parser.parse_args()

    if args.agent != GOLDEN and not args.base_url:
        parser.error(
            f"--agent {args.agent} is a real harness and needs --base-url (only {GOLDEN!r} runs without a model)"
        )
    base_url = args.base_url or "http://smoke.invalid/sessions/smoke"  # the golden agent never calls it
    benchmark = BENCHMARKS[args.benchmark]
    tasks_dir = args.tasks_dir if args.tasks_dir is not None else benchmark.tasks_dir()
    task = args.task or benchmark.smoke_task

    print(
        f"sandbox-smoke: connector={args.connector} backend={args.backend} agent={args.agent} "
        f"benchmark={args.benchmark} task={task}",
        flush=True,
    )
    result = asyncio.run(
        CONNECTORS[args.connector](tasks_dir, task, backend=args.backend, agent=args.agent, base_url=base_url)
    )
    print(f"sandbox-smoke: result={result}", flush=True)

    reward = float(result.get("reward", 0.0))
    exit_status = result.get("exit_status", "")
    if reward == 1.0 and exit_status == "Submitted":
        print("sandbox-smoke: PASS", flush=True)
        return 0
    print(f"sandbox-smoke: FAIL (reward={reward}, exit_status={exit_status!r})", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
