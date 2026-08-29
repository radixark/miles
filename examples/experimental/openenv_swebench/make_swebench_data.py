"""Generate a prompt dataset for the OpenEnv SWE-bench-style run.

The *tasks* are the SWE-Rebench-V2 "donor" variants: a directory tree (one per
task) that the env server serves at reset(). Each task dir contains a
``task.toml``, an ``instruction.md`` (served as the reset() instruction), an
``environment/`` (the image the env pulls), and a ``tests/`` verifier. This
builder only needs to emit, per task, the system prompt (how the agent should
behave) plus the ``task_id`` in metadata; ``swebench_agent_function`` drives the
multi-turn loop and reads metadata["task_id"].

task_ids are the top-level task directory names in the pool (they must equal the
dir name, since the env resolves a task by ``tasks_dir/<task_id>``); a valid task
dir contains a ``task.toml``.

    # all tasks in the pool
    python make_swebench_data.py --tasks_dir /root/swebench_pool \
        --output /root/swebench_train.jsonl
    # a small smoke subset
    python make_swebench_data.py --tasks_dir /root/swebench_pool \
        --output /root/swebench_smoke.jsonl --n 8
    # an explicit subset
    python make_swebench_data.py --tasks_dir /root/swebench_pool \
        --tasks synthdonor_99designs__aws-vault_8a6_738a5936
"""

import json
from pathlib import Path

from tap import Tap

# The agent contract must match swebench_agent_function._multi_turn: one shell
# command per turn inside a single ```bash block; TASK_COMPLETE to stop. The
# working directory is the repository root (the adapter cds the agent there).
_SYSTEM = (
    "You are an autonomous software engineer working inside a git repository. You "
    "will be given a task describing a change to make to the codebase, then "
    "interact with a real Linux shell whose working directory is the repository "
    "root. On each turn respond with EXACTLY ONE shell command inside a single "
    "```bash code block and nothing else. Explore the repository, implement the "
    "required change, and verify it (e.g. by building the project or running the "
    "relevant tests). Do not edit the test files that grade the task. When you are "
    "confident the task is fully complete, reply with TASK_COMPLETE (with no code "
    "block)."
)


class Args(Tap):
    tasks_dir: str = "/root/swebench_pool"  # pool of SWE-Rebench-V2 donor task dirs
    output: str = "/root/swebench_train.jsonl"
    n: int = 0  # 0 = all discovered tasks
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
        if args.n > 0:
            task_ids = task_ids[: args.n]

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
    print(f"Wrote {len(task_ids)} SWE-bench-style tasks to {args.output}")


if __name__ == "__main__":
    main()
