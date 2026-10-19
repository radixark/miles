"""The TMax Beaker head waits for its Ray job itself, not for `ray job submit`, against a scripted fake job client.

The fake answers list_jobs / get_job_status / get_job_info from a script of statuses (an exception stands for an
unreachable dashboard) and appends a line to the job's driver log on every poll, as a running driver would.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ray.job_submission import JobDetails, JobStatus, JobType

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax" / "beaker"))

import follow_ray_job  # noqa: E402

UNREACHABLE = ConnectionError("dashboard unreachable")
SUBMISSION_ID = "raysubmit_test"


class FakeJobClient:
    def __init__(self, *, first: JobStatus, then: list, log: Path):
        self.first, self.then, self.log, self.polls = first, list(then), log, 0

    def list_jobs(self):
        driver = JobDetails(type=JobType.DRIVER, status=JobStatus.SUCCEEDED, entrypoint="", start_time=9)
        job = JobDetails(
            type=JobType.SUBMISSION, submission_id=SUBMISSION_ID, status=self.first, entrypoint="train", start_time=5
        )
        return [driver, job]

    def get_job_status(self, submission_id):
        assert submission_id == SUBMISSION_ID
        self.polls += 1
        with self.log.open("a") as f:
            f.write(f"driver line {self.polls}\n")
        answer = self.then.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def get_job_info(self, submission_id):
        return SimpleNamespace(message=f"ended after {self.polls} polls")


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    log = tmp_path / "job-driver-{submission_id}.log"
    monkeypatch.setattr(follow_ray_job, "_DRIVER_LOG", str(log))
    monkeypatch.setattr(follow_ray_job.time, "sleep", lambda s: None)
    monkeypatch.setattr(follow_ray_job, "_LOG_OVERLAP_BYTES", 10)
    driver_log = Path(str(log).format(submission_id=SUBMISSION_ID))

    def run(*, first, then=(), log_before=None):
        if log_before is not None:
            driver_log.write_text(log_before)
        client = FakeJobClient(first=first, then=then, log=driver_log)
        monkeypatch.setattr(follow_ray_job, "JobSubmissionClient", lambda address: client)
        return follow_ray_job.main(), capsys.readouterr().out, client

    return run


def test_a_finished_job_is_taken_as_is(run):
    rc, out, client = run(first=JobStatus.SUCCEEDED)
    assert rc == 0 and client.polls == 0
    assert "ended SUCCEEDED" in out


@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.STOPPED])
def test_a_job_that_did_not_succeed_fails_the_replica(run, status):
    rc, out, _ = run(first=status)
    assert rc == 1 and f"ended {status.value}" in out


def test_a_running_job_is_followed_until_it_succeeds(run):
    rc, out, client = run(
        first=JobStatus.RUNNING,
        then=[JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.SUCCEEDED],
        log_before="printed before the stream dropped\nlast line\n",
    )
    assert rc == 0 and client.polls == 3
    assert "still RUNNING" in out
    # From the overlap before the old end (the last 10 bytes), then every line the driver wrote while followed.
    assert "printed before" not in out and "last line\n" in out
    assert [n for n in range(1, 4) if f"driver line {n}\n" in out] == [1, 2, 3]


def test_a_running_job_that_fails_later_fails_the_replica(run):
    rc, out, _ = run(first=JobStatus.PENDING, then=[JobStatus.RUNNING, JobStatus.FAILED])
    assert rc == 1 and "ended FAILED" in out


def test_an_unreachable_dashboard_is_waited_out(run):
    rc, out, client = run(first=JobStatus.RUNNING, then=[UNREACHABLE, UNREACHABLE, JobStatus.SUCCEEDED])
    assert rc == 0 and client.polls == 3
    assert out.count("cannot get the job's status") == 1  # reported once, not on every poll


def test_a_missing_driver_log_still_follows_the_status(run):
    rc, out, _ = run(first=JobStatus.RUNNING, then=[JobStatus.SUCCEEDED])
    assert rc == 0 and "not there (yet)" in out
    assert "driver line 1\n" in out  # the log appeared while followed, so it is printed from its start


def test_no_submitted_job_keeps_the_execute_result(run, monkeypatch):
    client = SimpleNamespace(list_jobs=lambda: [])
    monkeypatch.setattr(follow_ray_job, "JobSubmissionClient", lambda address: client)
    assert follow_ray_job.main() == 0
