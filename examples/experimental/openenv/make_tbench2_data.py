"""Generate a prompt dataset for the OpenEnv Terminal-Bench-2 (tbench2) run.

Unlike the coding/echo examples, the *tasks* are not hand-written: they are the
Terminal-Bench-2 suite shipped in the laude-institute/terminal-bench-2 repo. The
env serves the per-task instruction at reset(), so each prompt-data row only
needs the system prompt (how the agent should behave) plus the ``task_id`` in
metadata. ``openenv_agent_function`` (OPENENV_ENV_TYPE=tbench2) drives the
multi-turn loop and reads metadata["task_id"].

task_ids are the top-level task directory names in the TB2 repo checkout
(e.g. "chess-best-move"); a valid task dir contains a ``task.toml``.

    # all 89 tasks
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --output /root/tbench2_train.jsonl
    # a small smoke subset
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --output /root/tbench2_smoke.jsonl --n 8
    # an explicit subset
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --tasks chess-best-move,circuit-fibsqrt
"""

import json
from pathlib import Path

from tap import Tap

# The agent contract must match openenv_agent_function._multi_turn: one shell
# command per turn inside a single ```bash block; TASK_COMPLETE to stop.
_SYSTEM = (
    "You are an autonomous terminal agent solving a Terminal-Bench task. You will "
    "be given the task instruction, then interact with a real Linux shell. On each "
    "turn respond with EXACTLY ONE shell command inside a single ```bash code block "
    "and nothing else. Inspect the environment, make the required changes, and "
    "verify your work. When you are confident the task is fully complete, reply with "
    "TASK_COMPLETE (with no code block)."
)


class Args(Tap):
    tasks_dir: str = "/workspace/terminal-bench-2"  # TB2 repo checkout
    output: str = "/root/tbench2_train.jsonl"
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
    print(f"Wrote {len(task_ids)} TB2 tasks to {args.output}")


if __name__ == "__main__":
    main()
