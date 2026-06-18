"""Tests for the canonical CI label registry."""

from tests.ci.ci_register import register_cpu_ci
from tests.ci.labels import KNOWN_LABELS

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def test_known_labels_is_dict():
    assert isinstance(KNOWN_LABELS, dict)


def test_known_labels_initial_labels_present():
    expected = {
        "megatron",
        "model-scripts",
        "sglang",
        "fsdp",
        "short",
        "long",
        "ckpt",
        "lora",
        "precision",
        "weight-update",
    }
    assert expected <= set(KNOWN_LABELS), f"Missing canonical labels: {expected - set(KNOWN_LABELS)}"


def test_meta_labels_excluded():
    # Meta-labels (`run-ci-image`, `run-ci-all`) bypass the per-test labels
    # filter and must not be advertised as domain labels.
    assert "image" not in KNOWN_LABELS
    assert "all" not in KNOWN_LABELS


def test_descriptions_nonempty():
    for key, desc in KNOWN_LABELS.items():
        assert isinstance(desc, str) and desc.strip(), f"Label {key!r} has empty / non-string description: {desc!r}"
