"""Check that Clef training data files follow the record format in README.md.

Usage:
    python -m examples.experimental.clef.check_data --data-dir /data/clef [--model-dir /models/Qwen3.8-27B]

Pass ``--model-dir`` to also tokenize every record and check it fits ``--max-length``.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from examples.experimental.clef.data import encode_example, read_examples
from tap import Tap
from transformers import AutoProcessor

QUESTION_TYPES = ("choice", "noul", "score")


class Args(Tap):
    data_dir: str
    model_dir: str = ""
    max_length: int = 65536


def check_row(row: Any) -> list[str]:
    """Return human-readable format errors for one parsed JSONL row."""
    if not isinstance(row, dict):
        return ["row must be a JSON object"]
    errors = [f"missing top-level key {key!r}" for key in ("record", "targets", "source") if key not in row]
    if errors:
        return errors
    record, targets = row["record"], row["targets"]
    if not isinstance(row["source"], str) or not row["source"]:
        errors.append("source must be a non-empty string")
    if not isinstance(record, dict):
        return errors + ["record must be an object"]
    if not isinstance(record.get("id"), str) or not record["id"]:
        errors.append("record.id must be a non-empty string")
    if "state" not in record:
        errors.append("record.state is required")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        return errors + ["record.questions must be a non-empty object"]
    if not isinstance(targets, dict) or set(targets) != set(questions):
        return errors + ["targets must have exactly one entry per question"]
    for name, question in questions.items():
        where = f"question {name!r}"
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            errors.append(f"{where}: type must be one of {QUESTION_TYPES}")
            continue
        if not isinstance(question.get("instructions"), str) or not question["instructions"]:
            errors.append(f"{where}: instructions must be a non-empty string")
        criteria = question.get("criteria")
        if question["type"] == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                errors.append(f"{where}: choice criteria must map at least two option keys to descriptions")
                continue
            expected = set(map(str, criteria))
        elif question["type"] == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                errors.append(f"{where}: score criteria must list at least two ordered levels")
                continue
            expected = {str(index) for index in range(len(criteria))}
        else:
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                errors.append(f"{where}: noul criteria may only describe 'true' and 'false'")
            expected = {"true", "false"}
        target = targets[name]
        if not isinstance(target, dict) or set(map(str, target)) != expected:
            errors.append(f"{where}: target keys must be exactly {sorted(expected)}")
    return errors


def check_file(path: Path) -> list[str]:
    errors = []
    with path.open() as reader:
        for number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{path.name}:{number}: invalid JSON ({error.msg})")
                continue
            errors.extend(f"{path.name}:{number}: {message}" for message in check_row(row))
    return errors


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    root = Path(args.data_dir)
    paths = {split: root / f"{split}.jsonl" for split in ("train", "validation")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing data files: {missing}")
    errors = [error for path in paths.values() for error in check_file(path)]
    if errors:
        raise SystemExit("format errors:\n" + "\n".join(errors[:50]) + ("\n..." if len(errors) > 50 else ""))
    # Full validation shared with the trainer: target distributions, duplicate IDs.
    splits = {split: read_examples(path) for split, path in paths.items()}
    if not all(splits.values()):
        raise SystemExit("train.jsonl and validation.jsonl must both be non-empty")
    overlap = {e.record["id"] for e in splits["train"]} & {e.record["id"] for e in splits["validation"]}
    if overlap:
        raise SystemExit(f"{len(overlap)} record IDs appear in both splits, e.g. {sorted(overlap)[:5]}")
    summary = {
        split: {"records": len(examples), "sources": dict(Counter(e.source for e in examples))}
        for split, examples in splits.items()
    }
    if args.model_dir:
        tokenizer = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True).tokenizer
        lengths = [
            len(encode_example(tokenizer, e, args.max_length).encoded.input_ids)
            for examples in splits.values()
            for e in examples
        ]
        summary["max_tokens"] = max(lengths)
    print(json.dumps({"status": "ok", **summary}, indent=2))


if __name__ == "__main__":
    main()
