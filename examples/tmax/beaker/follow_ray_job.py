"""After the launcher's execute step returns 0, wait for the run's Ray job to actually end, and exit 0 only if it
succeeded.

`ray job submit` follows the job's log stream and exits 0 unless the job FAILED. So when the stream drops while the
job is still running, it returns 0 mid-run. On 2026-10-17 the dashboard on a starved head closed the stream, and the
entry script took training as finished (INVESTIGATIONS.md). This script asks Ray for the job's status instead. While
the job is pending or running, it keeps printing the job's driver log from about where the stream stopped, and waits
for the job to end.

    python examples/tmax/beaker/follow_ray_job.py

Exits 0 if the job SUCCEEDED (or no job was submitted) and 1 if it ended any other way. While Ray's dashboard cannot
be reached, it keeps waiting and reports every few minutes. It adds no time limit of its own: on a multi-node run, a
head whose Ray is gone makes the workers leave, and Beaker's propagateFailure then ends the experiment. It runs only
after the execute step returned 0, so Ray was up and had taken the job.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ray.job_submission import JobDetails, JobStatus, JobSubmissionClient, JobType

T = TypeVar("T")

_DASHBOARD = "http://127.0.0.1:8265"  # where run_ray_job submits the job
# Ray runs a job's driver on the head node (this one) and writes its output here, the same output the stream carries.
_DRIVER_LOG = "/tmp/ray/session_latest/logs/job-driver-{submission_id}.log"
# Reprint this much of the log before its current end: the stream lagged before it dropped, so its last lines may
# never have been printed. Repeated lines are harmless.
_LOG_OVERLAP_BYTES = 1 << 20
_POLL_S = 5.0
_REPORT_EVERY = 60  # polls between reports while Ray cannot be reached (about 5 minutes)


def main() -> int:
    client = _until_ok(lambda: JobSubmissionClient(_DASHBOARD), what="connecting to Ray's dashboard")
    job = _latest_submitted_job(client)
    if job is None:
        _say("no Ray job was submitted")
        return 0
    status = job.status
    if not status.is_terminal():
        _say(f"Ray job {job.submission_id} is still {status.value} although its submit returned; following it")
        status = _follow(client, submission_id=job.submission_id)
    message = _until_ok(lambda: client.get_job_info(job.submission_id).message, what="reading the job's end message")
    # Only its first line: a failed job's message goes on with the last 20,000 characters of the log printed above.
    first_line = message.splitlines()[0] if message else ""
    _say(f"Ray job {job.submission_id} ended {status.value}: {first_line}")
    return 0 if status == JobStatus.SUCCEEDED else 1


def _latest_submitted_job(client: JobSubmissionClient) -> JobDetails | None:
    jobs = _until_ok(client.list_jobs, what="listing Ray jobs")
    submitted = [job for job in jobs if job.type == JobType.SUBMISSION]
    return max(submitted, key=lambda job: job.start_time or 0, default=None)


def _follow(client: JobSubmissionClient, *, submission_id: str) -> JobStatus:
    """Print the driver log as it grows until the job reaches a terminal status, and return that status."""
    log = Path(_DRIVER_LOG.format(submission_id=submission_id))
    offset = _start_offset(log)
    misses = 0
    while True:
        offset = _print_new(log, offset=offset)
        try:
            status = client.get_job_status(submission_id)
        except Exception as e:  # a starved or restarting dashboard; the job itself may well still be running
            if misses % _REPORT_EVERY == 0:
                _say(f"cannot get the job's status ({misses} polls so far), still waiting: {e!r}")
            misses += 1
        else:
            misses = 0
            if status.is_terminal():
                _print_new(log, offset=offset)
                return status
        time.sleep(_POLL_S)


def _start_offset(log: Path) -> int:
    if not log.exists():
        _say(f"driver log {log} not there (yet); printing it if it appears")
        return 0
    offset = max(0, log.stat().st_size - _LOG_OVERLAP_BYTES)
    _say(f"printing the driver log from byte {offset}; its last {_LOG_OVERLAP_BYTES >> 20} MiB may repeat lines")
    return offset


def _print_new(log: Path, *, offset: int) -> int:
    if not log.exists():
        return offset
    with log.open("rb") as f:
        f.seek(offset)
        data = f.read()
    sys.stdout.buffer.write(data)
    sys.stdout.flush()
    return offset + len(data)


def _until_ok(call: Callable[[], T], *, what: str) -> T:
    misses = 0
    while True:
        try:
            return call()
        except Exception as e:  # Ray not reachable right now; it answers again once the head recovers
            if misses % _REPORT_EVERY == 0:
                _say(f"{what} failed ({misses} retries so far), still trying: {e!r}")
            misses += 1
            time.sleep(_POLL_S)


def _say(text: str) -> None:
    print(f"follow_ray_job: {text}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
