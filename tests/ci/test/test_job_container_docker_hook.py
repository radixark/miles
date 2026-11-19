"""Tests for tests/ci/github_runner/job-container-docker/docker, run as the runner would."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

HOOK_DIR = Path(__file__).resolve().parents[1] / "github_runner" / "job-container-docker"
SOCKET_MOUNT = "/var/run/docker.sock:/var/run/docker.sock"
# The job-container arguments the runner passes, as its `docker create` log line shows them.
RUNNER_CREATE = [
    "create",
    "--name",
    "job",
    "--gpus",
    "all",
    "-v",
    "/data/miles_ci/jit:/data/miles_ci/jit",
    "-v",
    SOCKET_MOUNT,
    "-v",
    "/data/runner-setup/data/r/_work:/__w",
    "--entrypoint",
    "tail",
    "radixark/miles:dev",
    "-f",
    "/dev/null",
]


def _fake_docker(directory: Path, calls: Path) -> None:
    directory.mkdir()
    docker = directory / "docker"
    docker.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {calls}\n')
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)


def _run(args: list[str], path_dirs: list[Path]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": ":".join(str(d) for d in path_dirs) + ":/usr/bin:/bin"}
    return subprocess.run([str(HOOK_DIR / "docker"), *args], env=env, capture_output=True, text=True)


@pytest.fixture
def next_docker(tmp_path):
    calls = tmp_path / "calls"
    _fake_docker(tmp_path / "next", calls)
    return tmp_path / "next", calls


@pytest.mark.parametrize("command", ["create", "run"])
def test_strips_only_the_socket_mount(next_docker, command):
    next_dir, calls = next_docker
    result = _run([command, *RUNNER_CREATE[1:]], [HOOK_DIR, next_dir])

    assert result.returncode == 0, result.stderr
    forwarded = calls.read_text().splitlines()
    assert SOCKET_MOUNT not in forwarded
    assert forwarded == [command, *_without_socket(RUNNER_CREATE[1:])]


def _without_socket(args: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(args):
        if args[i] == "-v" and args[i + 1] == SOCKET_MOUNT:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


@pytest.mark.parametrize("args", [["start", "job"], ["ps", "-a"], ["exec", "job", "sh", "-c", "true"]])
def test_other_commands_pass_through_unchanged(next_docker, args):
    next_dir, calls = next_docker
    result = _run(args, [HOOK_DIR, next_dir])

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == args


@pytest.mark.parametrize(
    "mount",
    [["--volume", SOCKET_MOUNT], ["-v", "/var/run/docker.sock:/sock"], ["--mount", "src=/var/run/docker.sock"]],
)
def test_any_other_socket_reference_fails_closed(next_docker, mount):
    next_dir, calls = next_docker
    result = _run(["create", *mount, "image"], [HOOK_DIR, next_dir])

    assert result.returncode == 1
    assert "refusing to pass the docker socket" in result.stderr
    assert not calls.exists()


def test_chains_to_the_next_docker_after_itself(tmp_path):
    calls = tmp_path / "calls"
    _fake_docker(tmp_path / "aliyun", calls)
    _fake_docker(tmp_path / "late", tmp_path / "late-calls")

    result = _run(RUNNER_CREATE, [HOOK_DIR, tmp_path / "aliyun", tmp_path / "late"])

    assert result.returncode == 0, result.stderr
    assert SOCKET_MOUNT not in calls.read_text().splitlines()
    assert not (tmp_path / "late-calls").exists()


def test_without_a_next_docker_fails(tmp_path):
    # Only bash on PATH: a hosted runner's /usr/bin also holds a real docker.
    (tmp_path / "bash").symlink_to(shutil.which("bash"))
    result = subprocess.run(
        [str(HOOK_DIR / "docker"), "ps"], env={"PATH": f"{HOOK_DIR}:{tmp_path}"}, capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "no docker after" in result.stderr


def test_never_execs_itself_through_another_path(tmp_path):
    (tmp_path / "bash").symlink_to(shutil.which("bash"))
    (tmp_path / "alias").symlink_to(HOOK_DIR)
    result = subprocess.run(
        [str(HOOK_DIR / "docker"), "ps"],
        env={"PATH": f"{HOOK_DIR}:{tmp_path / 'alias'}:{tmp_path}"},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert "no docker after" in result.stderr
