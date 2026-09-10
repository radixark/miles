"""Pressure search requires a passing N and an observed OOM at N+1."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from examples.multi_lora.pressure_test import PressureTest, _is_oom


@pytest.mark.parametrize("estimate,expected_trials", [(2, [2, 3, 4, 5]), (6, [6, 5, 4])])
def test_search_brackets_the_boundary_in_both_directions(tmp_path, estimate, expected_trials):
    report = tmp_path / "runs/probe-auto/slot-capacity.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"n_slots": estimate}))
    test = object.__new__(PressureTest)
    test.root = tmp_path
    test.args = Namespace(resume_auto=True)
    test.state = {"trials": []}
    test.run = SimpleNamespace(log=lambda record: None)
    test._record = lambda: None
    test._wait_ready = lambda trial: True
    test._launch = lambda trial, count: None
    test._stop_server = lambda trial: None
    counts = []

    def clients(trial, count):
        counts.append(count)
        return count <= 4

    test._clients = clients
    assert test.search() == 4
    assert counts == expected_trials


def test_only_explicit_cuda_oom_is_a_capacity_failure():
    assert _is_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")
    assert not _is_oom("Worker died. Possible causes include the OOM killer or SIGSEGV")
    assert not _is_oom("torch.OutOfMemoryError: DefaultCPUAllocator: can't allocate memory")
    assert not _is_oom("no free adapter slots (capacity 4)")
    assert not _is_oom("ReadTimeout: waiting for a response")


def test_late_oom_requires_new_lower_count_qualification():
    test = object.__new__(PressureTest)
    test.state = {"trials": []}
    test.run = SimpleNamespace(log=lambda record: None)
    test._record = lambda: None
    test._launch = lambda trial, count: None
    test._wait_ready = lambda trial: True
    test._clients = lambda trial, count: count == 2
    stopped = []
    test._stop_server = stopped.append
    assert test._qualify_below(4) == 2
    assert [trial["status"] for trial in test.state["trials"]] == ["oom", "passed"]
    assert stopped == ["late-oom-search-n3", "late-oom-search-n2"]


def test_stopped_ray_job_still_requires_process_cleanup(monkeypatch, tmp_path):
    test = object.__new__(PressureTest)
    terminal = SimpleNamespace(is_terminal=lambda: True)
    test.root = tmp_path
    test.args = Namespace(ray_address="head:6379")
    test._job = lambda trial: SimpleNamespace(status=terminal, submission_id="trial", job_id="12340000")
    test.jobs = SimpleNamespace(get_job_status=lambda _: terminal)

    def contaminated(*args, **kwargs):
        raise RuntimeError("old scheduler still alive")

    monkeypatch.setattr("examples.multi_lora.pressure_test.reap_trial_processes", contaminated)
    with pytest.raises(RuntimeError, match="old scheduler still alive"):
        test._stop_server("finished")
