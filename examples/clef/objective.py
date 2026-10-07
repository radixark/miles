"""Differentiable calibration loss and question-level evaluation metrics."""

import math
from collections import defaultdict
from collections.abc import Sequence

import torch

from examples.clef.data import LabeledRecord


def decision_loss(logits: list[list[torch.Tensor]], labels: Sequence[LabeledRecord]) -> torch.Tensor:
    # Average fields within each record, then records, so optional extra fields do
    # not change a question's weight or a distributed rank's effective sample count.
    record_losses = []
    for record_logits, label in zip(logits, labels, strict=True):
        losses = []
        for field_logits, target in zip(record_logits, label.targets, strict=True):
            probabilities = field_logits.float().softmax(-1)
            expected = torch.tensor(target, device=field_logits.device, dtype=torch.float32)
            losses.append((probabilities - expected).square().sum())
        record_losses.append(torch.stack(losses).mean())
    return torch.stack(record_losses).mean()


def prediction_rows(logits: list[list[torch.Tensor]], labels: Sequence[LabeledRecord]) -> list[dict]:
    rows = []
    for record_logits, label in zip(logits, labels, strict=True):
        for question, field_logits, target in zip(label.encoded.questions, record_logits, label.targets, strict=True):
            probabilities = field_logits.detach().float().softmax(-1).cpu().tolist()
            prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
            hard = max(target) == 1.0
            rows.append({
                "id": label.encoded.record_id,
                "field_id": question.question_id,
                "source": label.source,
                "option_ids": question.option_ids,
                "probabilities": probabilities,
                "target": target,
                "brier": sum((p - t) ** 2 for p, t in zip(probabilities, target, strict=True)),
                "confidence": max(probabilities),
                "collapsed": max(probabilities) == 1.0,
                "near_collapsed": max(probabilities) >= 0.999,
                "hard_target": hard,
                "correct": target[prediction] == 1.0 if hard else None,
            })
    return rows


def summarize(rows: Sequence[dict], bins: int = 15) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    records = defaultdict(list)
    for index, row in enumerate(rows):
        records[row.get("id", f"anonymous_{index}")].append(row["brier"])
    metrics = {
        # Match the loss's equal case weighting; also expose field-weighted Brier.
        "brier": sum(sum(values) / len(values) for values in records.values()) / len(records),
        "brier_per_field": sum(row["brier"] for row in rows) / len(rows),
        "collapse_percentage": 100 * sum(row["collapsed"] for row in rows) / len(rows),
        "near_collapse_percentage": 100 * sum(row["near_collapsed"] for row in rows) / len(rows),
        "questions": float(len(rows)),
        "records": float(len(records)),
    }
    hard = [row for row in rows if row["hard_target"]]
    if hard:
        metrics["single_answer_accuracy"] = sum(row["correct"] for row in hard) / len(hard)
        metrics["confidently_wrong_percentage"] = 100 * sum(
            not row["correct"] and row["confidence"] >= 0.9 for row in hard
        ) / len(hard)
        groups = defaultdict(list)
        for row in hard:
            groups[min(bins - 1, math.floor(row["confidence"] * bins))].append(row)
        metrics["ece"] = sum(
            abs(sum(row["confidence"] - row["correct"] for row in group)) for group in groups.values()
        ) / len(hard)
    return metrics
