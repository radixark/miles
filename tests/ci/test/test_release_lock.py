"""A release branch's lock names the sglang wheel its miles wheel depends on."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import setuptools
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
WHEEL = "0.5.22.post62+miles.g6136e6d2bd"


def test_the_lock_records_the_sglang_wheel(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/scripts/release/write_release_lock.py"),
            "release/v0.3.0",
            "6" * 40,
            WHEEL,
            "7" * 40,
            "release-v0.3.0-ci",
            "fp-x86",
            "fp-arm",
        ],
        cwd=tmp_path,
        check=True,
    )
    lock = json.loads((tmp_path / "release-lock.json").read_text())
    assert lock["sglang_commit"] == "6" * 40 and lock["sglang_wheel"] == WHEEL


def _install_requires(tmp_path, monkeypatch, lock):
    (tmp_path / "setup.py").write_text((ROOT / "setup.py").read_text())
    (tmp_path / "requirements.txt").write_text("ray\n")
    if lock is not None:
        (tmp_path / "release-lock.json").write_text(json.dumps(lock))
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(tmp_path)
    runpy.run_path("setup.py", run_name="__main__")
    return captured["install_requires"]


@pytest.mark.parametrize(
    "lock,pinned",
    [
        (None, []),
        ({"sglang_commit": "6" * 40, "sglang_wheel": WHEEL}, [f"sglang=={WHEEL}"]),
        ({"sglang_commit": "6" * 40}, []),
    ],
    ids=["main-has-no-lock", "release-branch", "cut-before-the-field"],
)
def test_setup_py_pins_sglang_only_from_a_lock_that_names_a_wheel(tmp_path, monkeypatch, lock, pinned):
    assert _install_requires(tmp_path, monkeypatch, lock) == ["ray", *pinned]
