import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.experimental.clef.check_data import check_row
from examples.experimental.clef.data import DecisionExample, LabeledRecord, augment_example, read_examples
from examples.experimental.clef.objective import decision_loss, prediction_rows, summarize
from examples.experimental.clef.train import _check_resume_config


def _label(targets: tuple[tuple[float, ...], ...], record_id: str = "r0") -> LabeledRecord:
    questions = [
        SimpleNamespace(question_id=f"q{i}", option_ids=[str(j) for j in range(len(t))]) for i, t in enumerate(targets)
    ]
    return LabeledRecord(SimpleNamespace(record_id=record_id, questions=questions), targets, "unit")


def _choice_example() -> DecisionExample:
    record = {
        "id": "coin",
        "state": "A biased coin lands heads with probability 0.7.",
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Guess the next flip.",
                "criteria": {"A": "Heads", "B": "Tails"},
            }
        },
    }
    return DecisionExample(record, {"answer": {"A": 0.7, "B": 0.3}}, "synthetic")


def test_decision_loss_is_brier_averaged_over_fields_then_records() -> None:
    logits = [[torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])], [torch.tensor([0.0, 0.0])]]
    labels = [_label(((1.0, 0.0), (0.0, 0.0, 1.0)), "a"), _label(((0.5, 0.5),), "b")]
    first = ((0.5 - 1) ** 2 + 0.5**2 + (2 * (1 / 3) ** 2 + (1 / 3 - 1) ** 2)) / 2
    expected = (first + 0.0) / 2
    assert decision_loss(logits, labels).item() == pytest.approx(expected)


def test_summarize_reports_accuracy_and_collapse() -> None:
    logits = [[torch.tensor([10.0, -10.0])], [torch.tensor([-10.0, 10.0])]]
    rows = prediction_rows(logits, [_label(((1.0, 0.0),), "a"), _label(((1.0, 0.0),), "b")])
    metrics = summarize(rows)
    assert metrics["single_answer_accuracy"] == 0.5
    assert metrics["confidently_wrong_percentage"] == 50.0
    assert metrics["near_collapse_percentage"] == 100.0
    assert metrics["records"] == 2.0


def test_augment_shuffles_options_together_with_targets() -> None:
    example = _choice_example()
    for seed in range(10):
        augmented = augment_example(example, random.Random(seed), multi_field_fraction=1.0)
        criteria = augmented.record["questions"]["answer"]["criteria"]
        target = augmented.targets["answer"]
        by_text = {criteria[key]: target[key] for key in criteria}
        assert by_text == {"Heads": 0.7, "Tails": 0.3}
        extra = augmented.record["questions"]["candidate_is_answer"]
        candidate = extra["instructions"].rsplit(" ", 1)[-1].rstrip(".")
        assert augmented.targets["candidate_is_answer"]["true"] == pytest.approx(target[candidate])


def test_read_examples_validates_targets(tmp_path: Path) -> None:
    example = _choice_example()
    row = {"record": example.record, "targets": example.targets, "source": example.source}
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(row) + "\n")
    assert read_examples(good) == [example]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**row, "targets": {"answer": {"A": 0.7, "B": 0.7}}}) + "\n")
    with pytest.raises(ValueError, match="sum to one"):
        read_examples(bad)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_examples(duplicate)


def test_check_row_accepts_all_question_types_and_reports_errors() -> None:
    example = _choice_example()
    row = {"record": example.record, "targets": example.targets, "source": example.source}
    assert check_row(row) == []

    multi = json.loads(json.dumps(row))
    multi["record"]["questions"]["urgent"] = {"type": "noul", "instructions": "Is it urgent?"}
    multi["record"]["questions"]["severity"] = {
        "type": "score",
        "instructions": "Rate it.",
        "criteria": ["low", "mid", "high"],
    }
    multi["targets"]["urgent"] = {"true": 0.2, "false": 0.8}
    multi["targets"]["severity"] = {"0": 0.1, "1": 0.6, "2": 0.3}
    assert check_row(multi) == []

    assert check_row({"record": example.record}) == [
        "missing top-level key 'targets'",
        "missing top-level key 'source'",
    ]
    wrong_keys = {**row, "targets": {"answer": {"A": 1.0, "C": 0.0}}}
    assert check_row(wrong_keys) == ["question 'answer': target keys must be exactly ['A', 'B']"]
    missing_target = json.loads(json.dumps(multi))
    del missing_target["targets"]["severity"]
    assert check_row(missing_target) == ["targets must have exactly one entry per question"]


def test_augment_leaves_named_fields_unchanged() -> None:
    record = {
        "id": "ticket",
        "state": "My card was charged twice.",
        "questions": {
            "team": {"type": "choice", "instructions": "Route it.", "criteria": {"billing": "Charges", "tech": "Bugs"}}
        },
    }
    example = DecisionExample(record, {"team": {"billing": 1.0, "tech": 0.0}}, "unit")
    assert augment_example(example, random.Random(0), multi_field_fraction=1.0) == example


def test_resume_allows_extension_but_not_changed_settings() -> None:
    from_checkpoint = {"total_steps": 100, "head_lr": 1e-5, "output_dir": "/a"}
    _check_resume_config(from_checkpoint, {"total_steps": 200, "head_lr": 1e-5, "output_dir": "/b"})
    with pytest.raises(ValueError, match="shorten"):
        _check_resume_config(from_checkpoint, {"total_steps": 50, "head_lr": 1e-5})
    with pytest.raises(ValueError, match="head_lr"):
        _check_resume_config(from_checkpoint, {"total_steps": 100, "head_lr": 2e-5})
