import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker"))

import image_inputs  # noqa: E402


def test_dockerfile_and_requirements_are_inputs():
    assert image_inputs._matches("docker/Dockerfile")
    assert image_inputs._matches("requirements.txt")
    assert image_inputs._matches("docker/patch/cu13/some.patch")


def test_rocm_dockerfile_is_not_a_cu13_input():
    """pr-test.yml builds the cu13 image only; ROCm has its own pipeline."""
    assert not image_inputs._matches("docker/Dockerfile.rocm")


def test_source_changes_are_not_image_inputs():
    assert not image_inputs._matches("miles/train.py")
    assert not image_inputs._matches("docker/README.md")


def test_compute_is_deterministic():
    assert image_inputs.compute() == image_inputs.compute()


def test_compute_tracks_every_declared_input(tmp_path, monkeypatch):
    """A file matching INPUT_GLOBS must move the hash, or rebuilds get skipped wrongly."""
    baseline = image_inputs.compute()
    monkeypatch.setattr(image_inputs, "_paths_at", lambda rev, root: ["requirements.txt"])
    monkeypatch.setattr(image_inputs, "_content_at", lambda path, rev, root: b"changed")
    assert image_inputs.compute() != baseline


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        (
            json.dumps(
                {
                    "linux/amd64": {"config": {"Labels": {image_inputs.LABEL_KEY: "abc"}}},
                    "linux/arm64": {"config": {"Labels": {image_inputs.LABEL_KEY: "abc"}}},
                }
            ),
            "abc",
        ),
        (json.dumps({"config": {"Labels": {image_inputs.LABEL_KEY: "def"}}}), "def"),
        (json.dumps({"linux/amd64": {"config": {"Labels": {"other": "x"}}}}), ""),
        ("", ""),
        ("not json", ""),
    ],
    ids=["multi-arch", "single-arch", "unlabelled", "unpublished", "malformed"],
)
def test_read_label(manifest, expected):
    assert image_inputs.read_label(manifest) == expected


@pytest.mark.parametrize(
    "error",
    [
        "429 Too Many Requests",
        "401 Unauthorized",
        "context deadline exceeded",
        "connection refused",
        "proxy.example: not found",
    ],
)
def test_registry_errors_do_not_request_a_rebuild(monkeypatch, error):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", error))
    with pytest.raises(RuntimeError, match="Image inspection failed"):
        image_inputs.inspect_published("radixark/miles:pr-3192")


def test_only_the_requested_missing_tag_is_rebuildable(monkeypatch):
    image = "radixark/miles:pr-3192"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", f"ERROR: {image}: not found\n"),
    )
    assert image_inputs.inspect_published(image) == ""
