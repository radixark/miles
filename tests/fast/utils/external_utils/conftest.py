import asyncio
import os
import signal
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from ray.job_submission import JobStatus

from miles.utils.external_utils import ray_job


@dataclass
class FakeJobSubmissionClient:
    status: JobStatus = JobStatus.SUCCEEDED
    submit_error: Exception | None = None
    log_chunks: list[str] = field(default_factory=list)
    signal_while_following: int | None = None
    addresses: list[str] = field(default_factory=list)
    submitted: list[dict[str, Any]] = field(default_factory=list)
    no_proxy_at_submit: list[str | None] = field(default_factory=list)
    tailed: list[str] = field(default_factory=list)
    status_queries: list[str] = field(default_factory=list)
    stop_commands: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def connect(self, address: str) -> "FakeJobSubmissionClient":
        self.addresses.append(address)
        return self

    def submit_job(self, *, submission_id: str, entrypoint: str, runtime_env: dict) -> str:
        self.no_proxy_at_submit.append(os.environ.get("no_proxy"))
        self.submitted.append({"submission_id": submission_id, "entrypoint": entrypoint, "runtime_env": runtime_env})
        if self.submit_error is not None:
            raise self.submit_error
        return submission_id

    async def tail_job_logs(self, submission_id: str) -> AsyncIterator[str]:
        self.tailed.append(submission_id)
        for chunk in self.log_chunks:
            yield chunk
        if self.signal_while_following is not None:
            signal.getsignal(self.signal_while_following)(self.signal_while_following, None)
            await asyncio.Event().wait()

    def get_job_status(self, submission_id: str) -> JobStatus:
        self.status_queries.append(submission_id)
        return self.status

    def get_job_info(self, submission_id: str) -> SimpleNamespace:
        return SimpleNamespace(message=f"the job {submission_id} broke")

    def stop(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.stop_commands.append((argv, kwargs))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    @property
    def stopped_ids(self) -> list[str]:
        return [argv[-1] for argv, _ in self.stop_commands]


@pytest.fixture
def fake_job_client(monkeypatch: pytest.MonkeyPatch) -> FakeJobSubmissionClient:
    client = FakeJobSubmissionClient()
    monkeypatch.setattr(ray_job, "JobSubmissionClient", client.connect)
    monkeypatch.setattr(ray_job.subprocess, "run", client.stop)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    return client


@pytest.fixture
def recorded_ray_submit_commands(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    commands: list[str] = []
    monkeypatch.setattr(ray_job, "exec_command_cpu", commands.append)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    return commands
