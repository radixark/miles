"""Convert calibration questions to labeled Clef schema records."""

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.clef.joint_schema_model import EncodedRecord, encode_record, question_options


@dataclass(frozen=True)
class DecisionExample:
    record: dict[str, Any]
    targets: dict[str, dict[str, float]]
    source: str


@dataclass(frozen=True)
class LabeledRecord:
    encoded: EncodedRecord
    targets: tuple[tuple[float, ...], ...]
    source: str


def validate_distribution(values: list[float]) -> None:
    if len(values) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("targets require at least two numeric probabilities")
    if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-8):
        raise ValueError("target probabilities must be finite, nonnegative, and sum to one")


def convert_row(row: Mapping[str, Any]) -> DecisionExample:
    metadata = row["metadata"]
    choices = metadata["choices"]
    target = metadata["target"]
    validate_distribution(target)
    if len(choices) != len(target) or not 2 <= len(choices) <= 26:
        raise ValueError("choice/target length mismatch or unsupported option count")
    options = {chr(65 + index): choice for index, choice in enumerate(choices)}
    # Remove the old JSON-report instruction without guessing where a question ends.
    option_block = "\n\n" + "\n".join(f"{key}. {text}" for key, text in options.items()).rstrip()
    content = row["prompt"][0]["content"]
    if content.count(option_block) != 1:
        raise ValueError(f"cannot unambiguously extract question {metadata['id']}")
    stem, suffix = content.split(option_block)
    if not suffix.startswith("\n\n") or not stem.strip():
        raise ValueError("unexpected calibration prompt format")
    record = {
        "id": metadata["id"],
        "state": stem,
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Choose the answer to the question in the state.",
                "criteria": options,
            }
        },
    }
    return DecisionExample(record, {"answer": dict(zip(options, target, strict=True))}, metadata["source"])


def read_examples(path: Path) -> list[DecisionExample]:
    with path.open() as reader:
        rows = [json.loads(line) for line in reader if line.strip()]
    examples = []
    for row in rows:
        if "record" in row:
            example = DecisionExample(row["record"], row["targets"], row["source"])
            if set(example.targets) != set(example.record["questions"]):
                raise ValueError("prepared schema/target fields mismatch")
            for field, target in example.targets.items():
                validate_distribution(list(target.values()))
                question = example.record["questions"][field]
                if question["type"] not in {"choice", "noul", "score"}:
                    raise ValueError("unsupported prepared question type")
                if {key for key, _ in question_options(question)} != set(target):
                    raise ValueError("prepared schema/target mismatch")
        else:
            example = convert_row(row)
        examples.append(example)
    ids = [example.record["id"] for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question IDs in {path}")
    return examples


def augment_example(example: DecisionExample, rng: random.Random, multi_field_fraction: float) -> DecisionExample:
    record = copy.deepcopy(example.record)
    targets = copy.deepcopy(example.targets)
    if "answer" not in record["questions"] or record["questions"]["answer"]["type"] != "choice":
        # Named workflow fields keep their identities; order is independent of labels.
        for question in record["questions"].values():
            if question["type"] == "choice":
                items = list(question["criteria"].items())
                rng.shuffle(items)
                question["criteria"] = dict(items)
        return DecisionExample(record, targets, example.source)
    options = record["questions"]["answer"]["criteria"]
    old_keys = list(options)
    rng.shuffle(old_keys)
    new_keys = [chr(65 + index) for index in range(len(options))]
    record["questions"]["answer"]["criteria"] = dict(zip(new_keys, [options[key] for key in old_keys], strict=True))
    targets["answer"] = dict(zip(new_keys, [example.targets["answer"][key] for key in old_keys], strict=True))
    if rng.random() < multi_field_fraction:
        candidate = rng.choice(new_keys)
        record["questions"]["candidate_is_answer"] = {
            "type": "noul",
            "instructions": f"The answer/outcome for the question in the state is option {candidate}.",
        }
        probability = targets["answer"][candidate]
        targets["candidate_is_answer"] = {"true": probability, "false": 1 - probability}
    return DecisionExample(record, targets, example.source)


def encode_example(tokenizer: Any, example: DecisionExample, max_length: int) -> LabeledRecord:
    # The upstream encoder truncates state at its limit. Training rejects oversized
    # records instead, so no ground-truth label is paired with silently lost evidence.
    encoded = encode_record(tokenizer, example.record, max_length=2**31 - 1)
    if len(encoded.input_ids) > max_length:
        raise ValueError(f"{encoded.record_id}: {len(encoded.input_ids)} tokens exceeds {max_length}")
    targets = tuple(
        tuple(example.targets[question.question_id][key] for key in question.option_ids)
        for question in encoded.questions
    )
    return LabeledRecord(encoded, targets, example.source)


def file_sha256(path: Path) -> str:
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()
