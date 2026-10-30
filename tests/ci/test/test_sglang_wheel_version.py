"""A release pins the sglang-miles wheel built from its locked commit."""

import importlib.util
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "release" / "sglang_wheel_version.py"
_spec = importlib.util.spec_from_file_location("sglang_wheel_version", _SCRIPT)
sglang_wheel_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sglang_wheel_version)

COMMIT = "6136e6d2bd4f0c1e9a7b5d3c2f1e0d9c8b7a6f5e"
RELEASES = [
    {
        "tag_name": "cu130-torch214-x86_64",
        "assets": [{"name": "sglang-9.9.9+miles.g6136e6d2bd-cp312-cp312-linux_x86_64.whl"}],
    },
    {
        "tag_name": "sglang-miles-2026w44-x86_64",
        "assets": [
            {"name": "sglang-source.json"},
            {"name": "sglang-0.5.22.post62+miles.g6136e6d2bd-cp311-cp311-manylinux_2_34_x86_64.whl"},
            {"name": "sglang-0.5.22.post62+miles.g6136e6d2bd-cp312-cp312-manylinux_2_34_x86_64.whl"},
        ],
    },
]


def test_the_version_comes_from_the_history_release_wheel_of_that_commit():
    assert sglang_wheel_version.wheel_version(COMMIT, RELEASES) == "0.5.22.post62+miles.g6136e6d2bd"


def test_a_commit_without_a_published_wheel_fails_the_cut():
    with pytest.raises(SystemExit, match="no sglang-miles wheel for 0000000000"):
        sglang_wheel_version.wheel_version("0" * 40, RELEASES)
