"""Tests for tests/ci/github_runner/remove-foreign-checkouts.sh, run as the job-start hook would."""

import os
import subprocess
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

HOOK = Path(__file__).resolve().parents[1] / "github_runner" / "remove-foreign-checkouts.sh"


def _run(work_dir: Path, repository: str) -> None:
    env = {**os.environ, "RUNNER_WORKSPACE": str(work_dir / repository.split("/")[1]), "GITHUB_REPOSITORY": repository}
    subprocess.run(["bash", str(HOOK)], env=env, check=True)


def test_keeps_only_the_starting_repository_and_runner_directories(tmp_path):
    work_dir = tmp_path / "_work"
    for name in ("miles", "other-repo", "_temp", "_actions"):
        (work_dir / name / "x").mkdir(parents=True)

    _run(work_dir, "radixark/miles")

    assert sorted(p.name for p in work_dir.iterdir()) == ["_actions", "_temp", "miles"]


def test_removes_a_symlinked_checkout_without_touching_its_target(tmp_path):
    work_dir, target = tmp_path / "_work", tmp_path / "shared"
    (work_dir / "miles").mkdir(parents=True)
    (target / "keep").mkdir(parents=True)
    (work_dir / "planted").symlink_to(target, target_is_directory=True)

    _run(work_dir, "radixark/miles")

    assert sorted(p.name for p in work_dir.iterdir()) == ["miles"]
    assert (target / "keep").is_dir()


def test_an_empty_work_directory_is_a_no_op(tmp_path):
    (tmp_path / "_work").mkdir()
    _run(tmp_path / "_work", "radixark/miles")
