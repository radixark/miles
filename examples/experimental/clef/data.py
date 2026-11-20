"""Load, augment, and encode labeled Clef schema records."""

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.experimental.clef.joint_schema_model import EncodedRecord, encode_record, question_options


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


def read_examples(path: Path) -> list[DecisionExample]:
    with path.open() as reader:
        rows = [json.loads(line) for line in reader if line.strip()]
    examples = []
    for row in rows:
        example = DecisionExample(row["record"], row["targets"], row["source"])
        if set(example.targets) != set(example.record["questions"]):
            raise ValueError("schema/target fields mismatch")
        for field, target in example.targets.items():
            validate_distribution(list(target.values()))
            question = example.record["questions"][field]
            if question["type"] not in {"choice", "noul", "score"}:
                raise ValueError("unsupported question type")
            if {key for key, _ in question_options(question)} != set(target):
                raise ValueError("schema/target option mismatch")
        examples.append(example)
    ids = [example.record["id"] for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question IDs in {path}")
    return examples


def augment_example(example: DecisionExample, rng: random.Random, multi_field_fraction: float) -> DecisionExample:
    record = copy.deepcopy(example.record)
    targets = copy.deepcopy(example.targets)
    if "answer" not in record["questions"] or record["questions"]["answer"]["type"] != "choice":
        # Named option keys are part of the input, and the encoder orders options
        # by key, so other fields keep their canonical order and are not augmented.
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
