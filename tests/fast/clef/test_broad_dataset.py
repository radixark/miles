import copy
import json
import random
import re
from pathlib import Path

import pytest

from examples.clef.build_broad_dataset import COUNTS, call_names, probability, workflow, workflow_bundle
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


def test_invoice_labels_follow_rendered_evidence() -> None:
    # Reconstruct the verdict from the actual shuffled documents, not generator locals.
    for index in range(200):
        row = workflow("invoice", index, "train", random.Random(index))
        state = row["record"]["state"]
        ordered = int(re.search(r"Purchase order PO-\S+: (\d+) units", state)[1])
        billed = int(re.search(r"Invoice INV-\S+: PO-\S+; (\d+) filters", state)[1])
        price = int(re.search(r"unit price (\d+)", state)[1])
        received = int(re.search(r"warehouse counted (\d+) accepted", state)[1])
        threshold = int(re.search(r"Amounts greater than (\d+) require", state)[1])
        paid = "settled last week, bank transaction reconciled" in state
        bank_ok = "bank change not requested" in state or "callback verification completed" in state
        approved = "I authorize payment of INV-" in state and "Cancel my payment approval" not in state
        blockers = [
            (paid, "duplicate"), (not bank_ok, "verify"), (billed > ordered, "correct"),
            (received < billed, "delivery"), (billed * price > threshold and not approved, "approve"),
        ]
        expected = next((action for blocked, action in blockers if blocked), "pay")
        assert row["targets"]["primary_action"][expected] == 1
        assert row["targets"]["pay_now"]["true"] == float(expected == "pay")


def test_long_bundle_scopes_every_field_and_preserves_targets() -> None:
    row = next(workflow_bundle("invoice", i, "train", random.Random(i)) for i in range(100)
               if workflow_bundle("invoice", i, "train", random.Random(i))["provenance"]["subcases"] == 20)
    assert len(row["record"]["questions"]) == 180
    for field, question in row["record"]["questions"].items():
        scope = "_".join(field.split("_")[:2])
        assert f"<case id='{scope}'>" in row["record"]["state"]
        assert question["instructions"].startswith(f"For {scope} only,")
        assert sum(row["targets"][field].values()) == 1


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
