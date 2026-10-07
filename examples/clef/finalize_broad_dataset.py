"""Select compact validation and audit a native decision fine-tuning dataset."""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from tap import Tap
from transformers import AutoTokenizer

from examples.clef.build_broad_dataset import audit
from examples.clef.data import file_sha256, read_examples


class Args(Tap):
    input_dir: str
    output_dir: str
    tokenizer_dir: str
    validation_size: int = 1024
    seed: int = 261006
    max_length: int = 65536


def select_validation(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if not 0 < size <= len(rows):
        raise ValueError("validation size must be positive and no larger than the source")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    counts = {source: len(group) * size // len(rows) for source, group in groups.items()}
    order = sorted(groups, key=lambda source: (-(len(groups[source]) * size % len(rows)), source))
    for source in order[:size - sum(counts.values())]:
        counts[source] += 1
    rng = random.Random(seed)
    selected = []
    for source in sorted(groups):
        selected.extend(rng.sample(groups[source], counts[source]))
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    source, output = Path(args.input_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("output directory already exists; preserve published versions")
    with (source / "validation.jsonl").open() as reader:
        selected = select_validation([json.loads(line) for line in reader], args.validation_size, args.seed)
    with (source / "train.jsonl").open() as reader:
        training = [json.loads(line) for line in reader]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    summary = audit({"train": training, "validation": selected}, tokenizer, args.max_length)
    output.mkdir(parents=True)
    shutil.copyfile(source / "train.jsonl", output / "train.jsonl")
    with (output / "validation.jsonl").open("w") as writer:
        for row in selected:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert file_sha256(output / "train.jsonl") == file_sha256(source / "train.jsonl")
    for split in summary:
        summary[split]["sha256"] = file_sha256(output / f"{split}.jsonl")
        assert len(read_examples(output / f"{split}.jsonl")) == summary[split]["records"]
    parent = json.loads((source / "manifest.json").read_text())
    manifest = {"splits": summary, "parent_manifest": parent, "selection": {
        "validation_size": args.validation_size, "seed": args.seed,
        "method": "source-stratified largest-remainder quotas, sampled without replacement",
        "training_sha256_unchanged": True, "max_length": args.max_length,
        "supervision": "native decision-head distributions; no generated-answer token loss",
    }}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(Path(__file__), output / "finalize_broad_dataset.py")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
