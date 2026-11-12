"""A release branch's lock names the sglang wheel its miles wheel pins."""

import json
import subprocess
import sys
from pathlib import Path

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
