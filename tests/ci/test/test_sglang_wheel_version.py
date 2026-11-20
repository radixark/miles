"""A release pins the sglang-miles wheel built from its locked commit, once every target has one."""

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
VERSION = "0.5.22.post62+miles.g6136e6d2bd"
PYTHONS = ["3.11", "3.12"]


def _wheel(python, arch, version=VERSION):
    return {"name": f"sglang-{version}-cp{python}-cp{python}-manylinux_2_34_{arch}.whl"}


def _releases(wheels):
    return [
        # the rolling image release is not the history: a wheel there does not count
        {"tag_name": "cu130-torch214-x86_64", "assets": [_wheel("312", "x86_64", "9.9.9+miles.g6136e6d2bd")]},
        {"tag_name": "sglang-miles-2026w44-x86_64", "assets": [{"name": "sglang-source.json"}, *wheels]},
    ]


ALL = [_wheel(python, arch) for python in ("311", "312") for arch in ("x86_64", "aarch64")]


def test_the_version_comes_from_the_history_release_wheels_of_that_commit():
    assert sglang_wheel_version.wheel_version(COMMIT, _releases(ALL), PYTHONS) == VERSION


@pytest.mark.parametrize(
    "wheels,missing",
    [([], "cp311 x86_64, cp311 aarch64, cp312 x86_64, cp312 aarch64"), (ALL[:3], "cp312 aarch64")],
    ids=["none-built", "one-target-failed"],
)
def test_a_commit_without_a_wheel_for_every_python_and_arch_fails_the_cut(wheels, missing):
    with pytest.raises(SystemExit, match=f"on {missing} in"):
        sglang_wheel_version.wheel_version(COMMIT, _releases(wheels), PYTHONS)


def test_the_pythons_are_the_ones_build_wheels_builds():
    assert sglang_wheel_version._pythons() == PYTHONS
