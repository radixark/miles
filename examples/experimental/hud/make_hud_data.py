"""Build a Miles prompt jsonl from a HUD v6 taskset.

A v6 task is a data row -- an env name, a template id, bound args -- minted by
calling a template in the env package's ``tasks.py``. This walks that module's
public ``tasks`` list and writes one Miles row per task (repeated ``--repeat``
times: GRPO needs groups, and a taskset like the browser starter's 2048 has a
handful of templates, not thousands of rows).

The ``prompt`` column is one chat message carrying the task slug: with a
chat-completions harness the model never sees this file -- its actual prompt
is the template's first yield, delivered inside the environment at run time.
Miles uses the column only to group samples of the same task (and its VLM data
path requires the chat-message shape, hence not a bare string).

    python -m examples.experimental.hud.make_hud_data \
        --env-dir /root/v6browser --task-ids 2048-score-5000 \
        --repeat 256 --output /root/hud2048_train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_tasks(env_dir: str | Path):
    """The env package's minted task rows (its public ``tasks`` list)."""
    env_dir = Path(env_dir).expanduser().resolve()
    sys.path.insert(0, str(env_dir))
    try:
        spec = importlib.util.spec_from_file_location("hud_env_tasks", env_dir / "tasks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.tasks)
    finally:
        sys.path.remove(str(env_dir))


def task_row(task) -> dict:
    """The part of a Task that travels to the rollout: enough to re-mint it."""
    return {"env": task.env, "id": task.id, "args": dict(task.args or {}), "slug": task.slug}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True, help="HUD env package (has tasks.py + Dockerfile.hud)")
    ap.add_argument("--task-ids", nargs="*", default=None, help="slugs to include; default all")
    ap.add_argument(
        "--args-json",
        default=None,
        help=(
            "JSON merged over each selected task's args, e.g. "
            "'{\"target_score\": 1024}' -- how a recipe retunes a template "
            "without editing the env package"
        ),
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--output", default="/root/hud_train.jsonl")
    args = ap.parse_args()

    tasks = load_tasks(args.env_dir)
    if args.task_ids:
        tasks = [t for t in tasks if t.slug in set(args.task_ids)]
    if not tasks:
        raise SystemExit(f"no tasks selected from {args.env_dir}")
    print(f"{len(tasks)} task(s): {[t.slug for t in tasks]}")

    overrides = json.loads(args.args_json) if args.args_json else {}
    written = 0
    with open(args.output, "w") as f:
        for _ in range(args.repeat):
            for t in tasks:
                row = task_row(t)
                row["args"] = {**row["args"], **overrides}
                record = {
                    "prompt": [{"role": "user", "content": t.slug}],
                    "label": "",
                    "metadata": {"hud_task": row},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
