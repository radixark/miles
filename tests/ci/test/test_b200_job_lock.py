import fcntl
import json
import select
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci
from tests.ci.github_runner.b200_job_lock import allocate_gpus, running_job_containers

register_cpu_ci(est_time=8, suite="stage-a-cpu", labels=[])

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Runner hooks use Linux pidfds and flock")


@pytest.fixture
def workers(tmp_path):
    processes = []

    def start(count):
        code = """
import json, os, sys
from pathlib import Path
from tests.ci.github_runner import b200_job_lock as locks
locks.running_job_containers = lambda root: {}
gpus = locks.acquire(os.getpid(), int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
print(json.dumps(gpus), flush=True)
sys.stdin.read()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(count), str(tmp_path), f"b200-oma-{count}gpu-{len(processes)}"],
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
    assert select.select([proc.stdout], [], [], 5)[0], "GPU allocation was not granted"
    line = proc.stdout.readline()
    assert line, proc.stderr.read()
    return set(json.loads(line))


def blocked(proc):
    assert not select.select([proc.stdout], [], [], 0.2)[0], "Overlapping GPU allocations were admitted"


def finish(proc):
    proc.communicate(timeout=5)
    assert proc.returncode == 0


def test_non_power_of_two_budgets_fill_the_host(workers):
    five = workers(5)
    five_gpus = acquired(five)
    three = workers(3)
    three_gpus = acquired(three)
    assert len(five_gpus) == 5 and len(three_gpus) == 3
    assert five_gpus.isdisjoint(three_gpus)
    assert five_gpus | three_gpus == set(range(8))
    finish(five)
    finish(three)


def test_mixed_counts_fill_host_and_reuse_only_released_gpus(workers):
    active = []
    used = set()
    for count in (4, 2, 1, 1):
        proc = workers(count)
        gpus = acquired(proc)
        assert len(gpus) == count and used.isdisjoint(gpus)
        used.update(gpus)
        active.append((proc, gpus))
    assert used == set(range(8))
    pending = workers(2)
    blocked(pending)
    finish(active[2][0])
    blocked(pending)
    finish(active[3][0])
    assert acquired(pending) == active[2][1] | active[3][1]
    whole = workers(8)
    blocked(whole)
    finish(active[0][0])
    finish(active[1][0])
    blocked(whole)
    finish(pending)
    assert acquired(whole) == set(range(8))
    finish(whole)


def test_noncontiguous_free_devices_can_form_a_larger_allocation(workers):
    active = [workers(2) for _ in range(4)]
    devices = [acquired(proc) for proc in active]
    assert len(set.union(*devices)) == 8
    finish(active[0])
    finish(active[2])
    larger = workers(4)
    assert acquired(larger) == devices[0] | devices[2]
    finish(larger)


def test_cancelled_holder_and_waiter_do_not_reserve_gpus(workers):
    whole = workers(8)
    acquired(whole)
    cancelled = workers(4)
    blocked(cancelled)
    cancelled.kill()
    cancelled.communicate(timeout=5)
    smaller = workers(2)
    blocked(smaller)
    whole.kill()
    whole.communicate(timeout=5)
    assert len(acquired(smaller)) == 2
    finish(smaller)
    replacement = workers(8)
    assert acquired(replacement) == set(range(8))
    finish(replacement)


def test_old_whole_node_job_excludes_new_allocator(workers, tmp_path):
    with (tmp_path / "b200-job.lock").open("a") as old_lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX)
        one = workers(1)
        blocked(one)
    assert len(acquired(one)) == 1
    finish(one)


def test_orphan_devices_stay_unavailable_until_container_cleanup(monkeypatch, tmp_path):
    containers = {}
    monkeypatch.setattr("tests.ci.github_runner.b200_job_lock.running_job_containers", lambda root: containers)
    with ExitStack() as old:
        gpus = allocate_gpus(tmp_path, 4, "b200-oma-4gpu-0", old)
    containers["b200-oma-4gpu-0"] = "orphan-container"
    with ExitStack() as live:
        other = allocate_gpus(tmp_path, 2, "b200-oma-2gpu-0", live)
        assert len(other) == 2 and set(other).isdisjoint(gpus)
        with pytest.raises(AssertionError, match="Orphaned B200 job containers"):
            allocate_gpus(tmp_path, 8, "b200-oma-8gpu-0", live)
        with pytest.raises(AssertionError, match="Previous job container remains"):
            allocate_gpus(tmp_path, 4, "b200-oma-4gpu-0", live)
    containers.clear()
    with ExitStack() as clean:
        assert allocate_gpus(tmp_path, 8, "b200-oma-8gpu-0", clean) == list(range(8))


def test_unknown_container_lease_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tests.ci.github_runner.b200_job_lock.running_job_containers",
        lambda root: {"b200-oma-2gpu-0": "unknown-container"},
    )
    with ExitStack() as stack, pytest.raises(AssertionError, match="has no GPU lease"):
        allocate_gpus(tmp_path, 1, "b200-oma-1gpu-0", stack)


def test_container_inventory_ignores_runner_and_nested_mounts(monkeypatch):
    root = Path("/data/miles_ci")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: (
            "runner /data/miles_ci,/var/run/docker.sock\n"
            "job /data/miles_ci/runner_b200-oma-1gpu-0/_temp,/data/miles_ci/runner_b200-oma-1gpu-0\n"
        ),
    )
    assert running_job_containers(root) == {"b200-oma-1gpu-0": "job"}
