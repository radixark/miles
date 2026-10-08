import select
import subprocess
import sys

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-cpu", labels=[])

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Runner hooks use Linux pidfds and flock")


@pytest.fixture
def workers(tmp_path):
    processes = []

    def start(count):
        code = """
import os, sys
from tests.ci.github_runner.b200_job_lock import acquire
acquire(os.getpid(), int(sys.argv[1]), sys.argv[2])
print('acquired', flush=True)
sys.stdin.read()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(count), str(tmp_path / "gpu.lock")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(proc)
        return proc

    yield start
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def acquired(proc):
    assert select.select([proc.stdout], [], [], 5)[0], "GPU lock was not granted"
    assert proc.stdout.readline() == b"acquired\n", proc.stderr.read()


def blocked(proc):
    assert not select.select([proc.stdout], [], [], 0.2)[0], "Overlapping GPU layouts acquired the lock"


def finish(proc):
    proc.communicate(timeout=5)
    assert proc.returncode == 0


def test_two_half_nodes_share_but_whole_node_excludes_both(workers):
    first, second = workers(4), workers(4)
    acquired(first)
    acquired(second)
    whole = workers(8)
    blocked(whole)
    finish(first)
    blocked(whole)
    finish(second)
    acquired(whole)
    half = workers(4)
    blocked(half)
    finish(whole)
    acquired(half)
    finish(half)


def test_cancelled_holder_and_waiter_release_their_locks(workers):
    whole = workers(8)
    acquired(whole)
    cancelled = workers(8)
    blocked(cancelled)
    cancelled.kill()
    cancelled.communicate(timeout=5)
    half = workers(4)
    blocked(half)
    whole.kill()
    whole.communicate(timeout=5)
    acquired(half)
    finish(half)


@pytest.mark.parametrize(
    ("gpu_count", "runner_name", "existing_runner", "blocked"),
    [
        (4, "b200-oma-4gpu-0", "b200-oma-8gpu-0", True),
        (4, "b200-oma-4gpu-0", "b200-oma-4gpu-0", True),
        (4, "b200-oma-4gpu-0", "b200-oma-4gpu-1", False),
        (4, "b200-oma-4gpu-1", "b200-oma-4gpu-0", False),
        (8, "b200-oma-8gpu-0", "b200-oma-4gpu-0", True),
        (8, "b200-oma-8gpu-0", "b200-oma-4gpu-1", True),
        (8, "b200-oma-8gpu-0", "b200-oma-8gpu-0", True),
    ],
)
def test_orphan_containers_block_only_overlapping_layouts(
    monkeypatch, gpu_count, runner_name, existing_runner, blocked
):
    from tests.ci.github_runner.b200_job_lock import assert_no_job_containers

    def docker_ps(command, *, text):
        assert command[:3] == ["docker", "ps", "--quiet"]
        return "orphan-job\n" if f"volume=/data/miles_ci/runner_{existing_runner}" in command else ""

    monkeypatch.setattr(subprocess, "check_output", docker_ps)
    if blocked:
        with pytest.raises(AssertionError, match="Overlapping B200 job containers remain.*orphan-job"):
            assert_no_job_containers(gpu_count, runner_name)
    else:
        assert_no_job_containers(gpu_count, runner_name)
