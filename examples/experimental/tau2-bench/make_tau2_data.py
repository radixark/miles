"""Write miles prompt data for tau2-bench: one jsonl per (domain, split).

``all_train.jsonl`` concatenates the train splits of every domain. Each row carries the task reference in ``metadata``; the tau2 agent function
loads the task itself, so ``prompt`` is only a readable id.

Usage:
    python make_tau2_data.py --output-dir /root/datasets/tau2
"""

import json
from pathlib import Path

import typer

from tau2.run import get_tasks

DOMAINS = ("airline", "retail", "telecom")
SPLITS = ("train", "test")


def main(output_dir: str = "/root/datasets/tau2"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_rows = []
    for domain in DOMAINS:
        for split in SPLITS:
            rows = [
                {
                    "prompt": f"{domain}/{task.id}",
                    "metadata": {"domain": domain, "task_split": split, "task_id": task.id},
                }
                for task in get_tasks(domain, task_split_name=split)
            ]
            _write_jsonl(Path(output_dir) / f"{domain}_{split}.jsonl", rows)
            if split == "train":
                train_rows += rows
    _write_jsonl(Path(output_dir) / "all_train.jsonl", train_rows)


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{path}: {len(rows)} tasks")


if __name__ == "__main__":
    typer.run(main)
