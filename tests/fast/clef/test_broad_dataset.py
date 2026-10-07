import copy
import json
import random
from pathlib import Path

import pytest

from examples.clef.build_broad_dataset import COUNTS, call_names, probability, workflow
from examples.clef.data import DecisionExample, augment_example, read_examples
from examples.clef.joint_schema_model import question_options


def test_curriculum_counts() -> None:
    assert sum(COUNTS["train"].values()) == 65536
    assert sum(COUNTS["validation"].values()) == 4096


def test_tool_labels_ignore_calls_in_argument_strings() -> None:
    assert call_names('[Web Search(query="text, Search(fake=1)"), other(items=[1, 2])]') == ["Web Search", "other"]
    assert call_names("[]") == []
    with pytest.raises(ValueError):
        call_names("[Search(x=1]")


def test_native_multifield_loader_and_augmentation(tmp_path: Path) -> None:
    row = workflow("invoice", 12, "train", random.Random(19))
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(row) + "\n")
    example = read_examples(path)[0]
    original = copy.deepcopy(example)
    augmented = augment_example(example, random.Random(13), 1)
    assert example == original
    assert augmented.targets == example.targets
    assert set(augmented.record["questions"]) == set(example.record["questions"])
    assert {q["type"] for q in example.record["questions"].values()} == {"choice", "noul", "score"}
    for key, question in augmented.record["questions"].items():
        assert {k for k, _ in question_options(question)} == set(augmented.targets[key])


@pytest.mark.parametrize("mutation", ["missing_target", "extra_target", "wrong_options", "negative_target"])
def test_malformed_multifield_records_rejected(tmp_path: Path, mutation: str) -> None:
    row = workflow("security", 3, "train", random.Random(21))
    if mutation == "missing_target":
        del row["targets"]["severity"]
    elif mutation == "extra_target":
        row["targets"]["extra"] = {"true": 1, "false": 0}
    elif mutation == "wrong_options":
        row["targets"]["severity"] = {"A": 1, "B": 0}
    else:
        row["targets"]["severity"] = {"0": -.1, "1": 1.1, "2": 0, "3": 0}
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        read_examples(path)


def test_policy_family_holdout_and_label_validity() -> None:
    for kind in ["invoice", "service", "security", "agent"]:
        for split in ["train", "validation"]:
            for i in range(100):
                row = workflow(kind, i, split, random.Random(i))
                family = row["provenance"]["policy_family"]
                assert (family < 6) == (split == "train")
                for field, values in row["targets"].items():
                    assert sum(values.values()) == 1
                    assert set(values.values()) <= {0, 1}
                    assert {k for k, _ in question_options(row["record"]["questions"][field])} == set(values)


def test_probabilistic_targets_and_variable_options() -> None:
    sizes = set()
    for i in range(60):
        row = probability(i, "train", random.Random(i))
        example = DecisionExample(row["record"], row["targets"], row["source"])
        augmented = augment_example(example, random.Random(i + 100), 1)
        target = augmented.targets["answer"]
        assert sum(target.values()) == pytest.approx(1)
        assert all(0 <= p <= 1 for p in target.values())
        assert augmented.targets["candidate_is_answer"]["true"] == target[
            augmented.record["questions"]["candidate_is_answer"]["instructions"].split("option ")[1][0]
        ]
        sizes.add(len(target))
    assert min(sizes) == 2 and max(sizes) > 10
