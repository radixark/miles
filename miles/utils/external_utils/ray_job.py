import asyncio
import os
import signal
import subprocess
import sys
import uuid
from contextlib import suppress

from ray.job_submission import JobStatus, JobSubmissionClient


def run_ray_job(*, address: str, entrypoint: str, runtime_env: dict) -> None:
    """Run a job until completion or a launcher signal, stopping it before returning."""
    runtime_env = {**runtime_env, "env_vars": {**runtime_env.get("env_vars", {}), "PYTHONUNBUFFERED": "1"}}
    previous_no_proxy = os.environ.get("no_proxy")
    os.environ["no_proxy"] = ",".join(
        filter(None, (previous_no_proxy or os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"))
    )
    try:
        asyncio.run(_run_ray_job(address, f"miles-{uuid.uuid4().hex}", entrypoint, runtime_env))
    finally:
        if previous_no_proxy is None:
            del os.environ["no_proxy"]
        else:
            os.environ["no_proxy"] = previous_no_proxy


async def _run_ray_job(address: str, submission_id: str, entrypoint: str, runtime_env: dict) -> None:
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    stop_signal = None

    def request_stop(signum, frame):
        nonlocal stop_signal
        stop_signal = signum
        loop.call_soon_threadsafe(stop_requested.set)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        client = JobSubmissionClient(address)
        try:
            client.submit_job(submission_id=submission_id, entrypoint=entrypoint, runtime_env=runtime_env)
            print(f"Ray job {submission_id} submitted to {address}", flush=True)
            await _follow_job(client, submission_id, stop_requested)
        finally:
            # Unlike the stop HTTP response, the CLI waits for a terminal job status.
            subprocess.run(
                [sys.executable, "-m", "ray.scripts.scripts", "job", "stop", "--address", address, submission_id],
                check=True,
                timeout=45,
            )
        if stop_signal is not None:
            raise SystemExit(128 + stop_signal)
        status = client.get_job_status(submission_id)
        if status != JobStatus.SUCCEEDED:
            raise RuntimeError(
                f"Ray job {submission_id} ended with {status}: {client.get_job_info(submission_id).message}"
            )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


async def _follow_job(client: JobSubmissionClient, submission_id: str, stop_requested: asyncio.Event) -> None:
    async def print_logs():
        async for chunk in client.tail_job_logs(submission_id):
            print(chunk, end="", flush=True)

    logs_task = asyncio.create_task(print_logs())
    stop_task = asyncio.create_task(stop_requested.wait())
    try:
        done, _ = await asyncio.wait((logs_task, stop_task), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (logs_task, stop_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
