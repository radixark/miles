"""Generate disjoint, deduplicated Reasoning Gym puzzle datasets."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.mopd_puzzles.tasks import SYSTEM_PROMPT, score
from reasoning_gym import create_dataset

CONFIGS = {
    "countdown4": ("countdown", dict(min_numbers=4, max_numbers=4, max_value=20, min_target=10, max_target=100)),
    "countdown5": ("countdown", dict(min_numbers=5, max_numbers=5, max_value=30, min_target=30, max_target=150)),
    "countdown6": ("countdown", dict(min_numbers=6, max_numbers=6, max_value=50, min_target=50, max_target=250)),
    "graph8": ("graph_color", dict(min_num_vertices=8, max_num_vertices=8, edge_probability=0.2)),
    "graph12": ("graph_color", dict(min_num_vertices=12, max_num_vertices=12, edge_probability=0.2)),
    "graph16": ("graph_color", dict(min_num_vertices=16, max_num_vertices=16, edge_probability=0.2)),
    "graph24": ("graph_color", dict(min_num_vertices=24, max_num_vertices=24, edge_probability=0.2)),
}
SPLIT_SEEDS = {"screen": 1000000, "train": 10000000, "dev": 20000000, "test": 30000000}


def _row(entry, domain, config_name, split, index):
    metadata = entry["metadata"]
    if domain == "countdown":
        label = dict(domain=domain, numbers=metadata["numbers"], target=metadata["target"])
        problem = (
            f"Use all these numbers exactly once: {label['numbers']}. Make {label['target']}. "
            "Use only +, -, *, / and parentheses. Return an arithmetic expression in the answer block."
        )
        oracle = entry["answer"]
        canonical = dict(domain=domain, numbers=sorted(label["numbers"]), target=label["target"])
    else:
        label = dict(domain=domain, puzzle=metadata["puzzle"])
        problem = entry["question"].split("Return your solution")[0]
        problem += "Return a JSON object mapping every vertex to its integer color in the answer block."
        oracle = json.dumps(metadata["possible_answer"])
        canonical = label
    identity = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    # Validate generator answers with the same strict verifier used for training.
    if score(f"<answer>{oracle}</answer>", label) != 1:
        return None
    return dict(
        prompt=[dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=problem)],
        label=json.dumps(label),
        metadata=dict(
            opd_teacher=domain, domain=domain, config=config_name, split=split, source_index=index, puzzle_id=identity
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--splits", nargs="+", default=["screen"])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    # Reuse existing identities to prevent overlap when generating later splits.
    for path in sorted(args.output.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            seen.add(json.loads(line)["metadata"]["puzzle_id"])
    for split in args.splits:
        for config_name in args.configs:
            path = args.output / f"{config_name}-{split}.jsonl"
            if path.exists():
                raise FileExistsError(path)
            domain, config = CONFIGS[config_name]
            config_index = list(CONFIGS).index(config_name)
            dataset = create_dataset(domain, seed=SPLIT_SEEDS[split] + config_index * 100000, size=100000, **config)
            rows = []
            for index in range(100000):
                row = _row(dataset[index], domain, config_name, split, index)
                if row is None or row["metadata"]["puzzle_id"] in seen:
                    continue
                seen.add(row["metadata"]["puzzle_id"])
                rows.append(row)
                if len(rows) == args.size:
                    break
            if len(rows) != args.size:
                raise RuntimeError(f"Not enough valid unique examples for {config_name}")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            print(path, len(rows), flush=True)


if __name__ == "__main__":
    main()
