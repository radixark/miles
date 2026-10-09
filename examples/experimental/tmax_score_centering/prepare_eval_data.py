"""Download terminal-bench@2.0 and prepare evaluation-only Miles conversations.

The Harbor registry selects the tasks and pins their Git revisions. Only the
task instruction enters the prompt; verifiers and solutions remain in Harbor
task directories. The manifest beside the JSONL records source task identities.

Example:
    python -m examples.experimental.tmax_score_centering.prepare_eval_data \
        --output /data/tmax/eval/terminal-bench-2.0.jsonl \
        --harbor-tasks-dir /data/tmax/tasks
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import yaml

DATASET = "terminal-bench@2.0"
_PROMPTS = yaml.safe_load(Path(__file__).with_name("vanillux_prompts.yaml").read_text())


async def _download_dataset() -> tuple[dict, dict[str, Path]]:
    # Harbor is needed only for downloading; local conversion can run on a laptop.
    from harbor.registry.client.factory import RegistryClientFactory

    client = RegistryClientFactory.create(
        registry_url="https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
    )
    downloaded = await client.download_dataset(name=DATASET)
    spec = {
        "name": "terminal-bench",
        "version": "2.0",
        "tasks": [{"name": item.id.get_name(), **item.id.model_dump(mode="json")} for item in downloaded],
    }
    return spec, {item.id.get_name(): item.downloaded_path for item in downloaded}


def prepare(spec: dict, task_paths: dict[str, Path], output_path: Path, harbor_tasks_dir: Path) -> int:
    if f"{spec['name']}@{spec['version']}" != DATASET:
        raise ValueError(f"expected the Harbor registry entry for {DATASET}")
    names = [task["name"] for task in spec["tasks"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("evaluation task identities must be nonempty and unique")
    records = []
    for task in spec["tasks"]:
        source = task_paths[task["name"]].resolve()
        for required in ("instruction.md", "task.toml", "environment", "tests"):
            if not (source / required).exists():
                raise FileNotFoundError(source / required)
        instance_id = f"terminal-bench-2.0__{task['name']}"
        destination = (harbor_tasks_dir / instance_id).resolve()
        if harbor_tasks_dir.resolve() not in destination.parents:
            raise ValueError(f"invalid task name: {task['name']!r}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        instruction = (source / "instruction.md").read_text().strip()
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": _PROMPTS["system_template"]},
                    {"role": "user", "content": _PROMPTS["instance_template"].replace("{{task}}", instruction)},
                ],
                "metadata": {
                    "instance_id": instance_id,
                    "task_id": task["name"],
                    "dataset": DATASET,
                    "split": "eval",
                    "git_url": task["git_url"],
                    "git_commit_id": task["git_commit_id"],
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    output_path.with_suffix(".manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harbor-tasks-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, help="Use a local checkout instead of downloading")
    parser.add_argument("--registry-entry", type=Path, help="Registry DatasetSpec JSON for the local checkout")
    args = parser.parse_args()
    if (args.task_dir is None) != (args.registry_entry is None):
        parser.error("--task-dir and --registry-entry must be supplied together")
    if args.task_dir is None:
        spec, paths = asyncio.run(_download_dataset())
    else:
        spec = json.loads(args.registry_entry.read_text())
        paths = {task["name"]: args.task_dir / task["path"] for task in spec["tasks"]}
    count = prepare(spec, paths, args.output, args.harbor_tasks_dir)
    print(f"wrote {count} {DATASET} evaluation tasks to {args.output}")


if __name__ == "__main__":
    main()
