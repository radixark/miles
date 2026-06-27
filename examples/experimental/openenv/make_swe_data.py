"""Generate a prompt dataset for the OpenEnv SWE-bench run.

Like the tbench2 example, the *tasks* are not hand-written: they are harbor-format
SWE-bench task dirs (one per instance_id, e.g. "astropy__astropy-12907"), each
containing a ``task.toml``, an ``environment/Dockerfile`` (FROM the swebench eval
image), ``instruction.md`` (the problem statement), and ``tests/`` (the swebench
verifier harness). The swe_env serves instruction.md at reset(), so each
prompt-data row only needs the system prompt plus the ``task_id`` (instance_id)
in metadata. ``openenv_agent_function`` (OPENENV_ENV_TYPE=swe) drives the
multi-turn loop and reads metadata["task_id"].

    # randomly sample 100 tasks (reproducible via --seed)
    python make_swe_data.py --tasks_dir /home/ubuntu/harbor_tasks_swebench_verified \
        --output /root/swe_train.jsonl --n 100 --seed 0
    # all discovered tasks
    python make_swe_data.py --tasks_dir /home/ubuntu/harbor_tasks_swebench_verified \
        --output /root/swe_all.jsonl
    # an explicit subset
    python make_swe_data.py --tasks_dir /home/ubuntu/harbor_tasks_swebench_verified \
        --tasks astropy__astropy-12907,sympy__sympy-13647
"""

import json
import random
from pathlib import Path

from tap import Tap

# The agent contract must match openenv_agent_function._multi_turn: one shell
# command per turn inside a single ```bash block; TASK_COMPLETE to stop. The repo
# is already checked out at /testbed inside the container.
_SYSTEM = (
    "You are an autonomous software engineering agent fixing a bug in a real "
    "Python repository. The repository is checked out at /testbed and the issue "
    "to resolve is given as the task instruction. On each turn respond with "
    "EXACTLY ONE shell command inside a single ```bash code block and nothing "
    "else. Explore the codebase, edit the source to resolve the issue, and verify "
    "your change. Do NOT edit the test files. When you are confident the fix is "
    "complete, reply with TASK_COMPLETE (with no code block)."
)


class Args(Tap):
    tasks_dir: str = "/home/ubuntu/harbor_tasks_swebench_verified"  # SWE task dirs
    output: str = "/root/swe_train.jsonl"
    n: int = 100  # 0 = all discovered tasks
    seed: int = 0  # RNG seed for the random sample
    tasks: str = ""  # optional comma-separated explicit task_ids


def _discover_task_ids(tasks_dir: Path) -> list[str]:
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())


def main() -> None:
    args = Args().parse_args()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = _discover_task_ids(tasks_dir)
        if args.n > 0 and args.n < len(task_ids):
            task_ids = sorted(random.Random(args.seed).sample(task_ids, args.n))

    missing = [t for t in task_ids if not (tasks_dir / t / "task.toml").is_file()]
    if missing:
        raise SystemExit(f"task_ids without a task.toml in {tasks_dir}: {missing}")

    with open(args.output, "w") as f:
        for tid in task_ids:
            row = {
                "prompt": [{"role": "system", "content": _SYSTEM}],
                "metadata": {"task_id": tid},
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(task_ids)} SWE-bench tasks to {args.output}")


if __name__ == "__main__":
    main()
