"""Prepare TMax Open-Instruct rows for a Miles score-centering run.

The released TMax training rows contain the policy conversation in ``messages``
and the sandbox identity in ``env_config``.  Miles keeps those as ``prompt`` and
``metadata`` respectively.  Verifier truth and raw task-generation fields are
intentionally not copied into the training rows.

Example::

    python examples/experimental/tmax_score_centering/prepare_data.py \
        --input /data/tmax-15k-open-instruct/data/train-00000-of-00001.parquet \
        --output /data/tmax-15k-miles-grpo.jsonl \
        --task-dir /data/tmax-tasks

The resulting file is consumed with ``--prompt-data`` and
``--input-key prompt --metadata-key metadata``. Do not use
``--apply-chat-template``: the agent passes messages to the Miles session
server, which owns chat templating and tokenization.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Open-Instruct Parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Miles JSONL output path")
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Directory containing one extracted task directory per TMax task ID",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke dataset")
    parser.add_argument("--task-id", action="append", help="Select a released task ID (repeatable)")
    parser.add_argument(
        "--harbor-tasks-dir", type=Path, help="Materialize Harbor task directories using released images"
    )
    return parser.parse_args()


def _metadata(row: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    env_config = row["env_config"]
    task_id = str(env_config["task_id"])
    resolved_task_dir = (task_dir / task_id).resolve()
    if task_dir.resolve() not in resolved_task_dir.parents:
        raise ValueError(f"task id escapes task directory: {task_id!r}")
    if not resolved_task_dir.is_dir():
        raise FileNotFoundError(f"missing extracted task directory for {task_id!r}: {resolved_task_dir}")

    return {
        "instance_id": task_id,
        "task_id": task_id,
        "image": str(env_config["image"]),
        "env_name": str(env_config["env_name"]),
        "source": str(row["source"]),
        "dataset": str(row["dataset"]),
        "task_dir": str(resolved_task_dir),
    }


def _write_harbor_task(source: Path, destination: Path, image: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "instruction.md", destination / "instruction.md")
    shutil.copytree(source / "tests", destination / "tests", dirs_exist_ok=True)
    (destination / "environment").mkdir(exist_ok=True)
    # Task setup and fixtures are already baked into the released image.
    # Harbor uploads tests only when verification begins.
    (destination / "task.toml").write_text(
        'schema_version = "1.1"\n[agent]\ntimeout_sec = 900\n'
        "[verifier]\ntimeout_sec = 600\n[environment]\n"
        f"docker_image = {json.dumps(image)}\nbuild_timeout_sec = 600\n"
        "cpus = 1\nmemory_mb = 2048\nstorage_mb = 10240\ngpus = 0\nallow_internet = true\n"
    )


def prepare(
    input_path: Path,
    output_path: Path,
    task_dir: Path,
    limit: int | None,
    task_ids: list[str] | None = None,
    harbor_tasks_dir: Path | None = None,
) -> int:
    rows = parquet.read_table(input_path).to_pylist()
    if task_ids is not None:
        selected = set(task_ids)
        rows = [row for row in rows if row["env_config"]["task_id"] in selected]
        missing = selected - {row["env_config"]["task_id"] for row in rows}
        if missing:
            raise ValueError(f"unknown task IDs: {sorted(missing)}")
    selected_rows = rows if limit is None else rows[:limit]
    if not selected_rows:
        raise ValueError("input dataset is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in selected_rows:
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("each TMax row must contain a non-empty messages list")
            if [message["role"] for message in messages] != ["system", "user"]:
                raise ValueError("TMax rows must contain exactly one system message followed by one user message")
            record = {"prompt": messages, "metadata": _metadata(row, task_dir)}
            if harbor_tasks_dir is not None:
                meta = record["metadata"]
                _write_harbor_task(Path(meta["task_dir"]), harbor_tasks_dir / meta["task_id"], meta["image"])
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(selected_rows)


def main() -> None:
    args = _parse_args()
    count = prepare(args.input, args.output, args.task_dir, args.limit, args.task_id, args.harbor_tasks_dir)
    print(f"wrote {count} TMax rows to {args.output}")


if __name__ == "__main__":
    main()
