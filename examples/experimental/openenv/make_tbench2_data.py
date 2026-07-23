"""Generate a prompt dataset for the OpenEnv Terminal-Bench-2 (tbench2) run.

The *tasks* are not hand-written: they are the
Terminal-Bench-2 suite shipped in the laude-institute/terminal-bench-2 repo. The
env serves the per-task instruction at reset(), so each prompt-data row only
needs the system prompt (how the agent should behave) plus the ``task_id`` in
metadata. ``openenv_agent_function`` drives the multi-turn loop and reads
metadata["task_id"].

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

# The agent contract (one ```bash block per turn; TASK_COMPLETE to stop) is
# defined by the adapter, next to the code that parses it.
from openenv_agent_function import TB2_AGENT_SYSTEM_PROMPT
from tap import Tap


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
                "prompt": [{"role": "system", "content": TB2_AGENT_SYSTEM_PROMPT}],
                "metadata": {"task_id": tid},
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(task_ids)} TB2 tasks to {args.output}")


if __name__ == "__main__":
    main()
