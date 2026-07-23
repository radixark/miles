"""Tests for the per-task sandbox recipe's verifier-asset hygiene.

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the recipe:

    pytest examples/experimental/openenv/tests/ -q
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tb2_sandbox_recipe as recipe  # noqa: E402


def _make_tasks_repo(root: Path, task_name: str = "some-task") -> Path:
    """A minimal tasks checkout: git repo with a GitHub origin and one task."""
    repo = root / "tb2repo"
    task = repo / task_name
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    for cmd in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", "https://github.com/acme/tb2tasks.git"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@e.st", "-c", "user.name=t", "commit", "-qm", "x"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return task


def test_task_layer_excludes_solution(tmp_path: Path):
    task = _make_tasks_repo(tmp_path)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=task.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    cmd = recipe._task_layer_command(task)

    # The exclusion is anchored to this one task's solution/ (dir + contents),
    # not a loose pattern that could drop task files elsewhere.
    assert f"--exclude='tb2tasks-{sha}/some-task/solution'" in cmd
    assert f"--exclude='tb2tasks-{sha}/some-task/solution/*'" in cmd
    assert cmd.rstrip().endswith(f"'tb2tasks-{sha}/some-task'")


def test_dir_tar_b64_is_deterministic(tmp_path: Path):
    # The b64 is embedded in a build command: identical source must yield
    # identical bytes, or every call produces a new image definition and
    # provider build caches / pre-baked snapshots never hit. Guards the two
    # nondeterminism sources: the gzip header timestamp (differs between
    # calls in different seconds) and per-entry mtime/owner metadata
    # (differs between checkouts of the same content).
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    sub = src / "server"
    sub.mkdir()
    (sub / "b.py").write_text("y = 2\n")

    first = recipe._dir_tar_b64([src], ["pkg"], max_bytes=100_000)
    time.sleep(1.1)  # cross a gzip-header timestamp boundary
    os.utime(src / "a.py", (0, 0))  # same content, different file mtime
    second = recipe._dir_tar_b64([src], ["pkg"], max_bytes=100_000)

    assert first == second


def test_server_cmd_sets_withhold_gate():
    assert "TB2_WITHHOLD_TESTS=1" in recipe.server_cmd()


def test_server_cmd_defaults_to_the_staged_task():
    # A per-task sandbox stages one task; a reset() with no task_id must land
    # on it, not the env's built-in headless-terminal default.
    cmd = recipe.server_cmd(default_task_id="fix-git")
    assert "TB2_DEFAULT_TASK_ID=fix-git " in cmd
