"""Tests for RayMainJob stop-failure reconciliation.

Previously, _stop_locked() did not clean up _job_id when _stop_job() threw
an exception (e.g. timeout). This left _job_id non-None even though the
remote job had already terminated, causing _start_locked() to permanently
reject new submissions with "previous job still tracked".
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob
from miles.utils.ft.adapters.types import JobStatus


def _make_job(
    *,
    get_status_response: str = "STOPPED",
    stop_side_effect: Exception | None = None,
    submit_response: str = "new_job_id",
) -> RayMainJob:
    client = MagicMock()
    client.get_job_status.return_value = get_status_response
    client.stop_job.side_effect = stop_side_effect
    client.submit_job.return_value = submit_response

    job = RayMainJob(
        client=client,
        entrypoint="echo hello",
        ft_id="test",
        k8s_label_prefix="test",
        poll_interval_seconds=0.01,
        submit_timeout_seconds=5,
        get_status_timeout_seconds=5,
        stop_job_timeout_seconds=0.01,
    )
    return job


class TestStopFailureReconciliation:
    @pytest.mark.asyncio
    async def test_stop_timeout_clears_job_id_when_remote_already_stopped(self) -> None:
        """_stop_job() times out but the remote job is already in STOPPED state.
        After reconciliation, _job_id should be cleared so start() can proceed."""
        job = _make_job(get_status_response="STOPPED")
        job._job_id = "old_job"

        with pytest.raises(TimeoutError):
            await job.stop(timeout_seconds=0)

        # Reconciliation should have cleared _job_id because remote is STOPPED
        assert job.job_id is None

    @pytest.mark.asyncio
    async def test_stop_timeout_clears_job_id_when_remote_failed(self) -> None:
        """Remote job is FAILED — _job_id should still be cleared."""
        job = _make_job(get_status_response="FAILED")
        job._job_id = "old_job"

        with pytest.raises(TimeoutError):
            await job.stop(timeout_seconds=0)

        assert job.job_id is None

    @pytest.mark.asyncio
    async def test_stop_timeout_keeps_job_id_when_remote_still_running(self) -> None:
        """Remote job is still RUNNING — _job_id must NOT be cleared."""
        job = _make_job(get_status_response="RUNNING")
        job._job_id = "old_job"

        with pytest.raises(TimeoutError):
            await job.stop(timeout_seconds=0)

        assert job.job_id == "old_job"

    @pytest.mark.asyncio
    async def test_start_succeeds_after_reconciled_stop_timeout(self) -> None:
        """Full scenario: stop times out, reconciliation clears stale _job_id,
        subsequent start() succeeds instead of raising 'still tracked'."""
        job = _make_job(get_status_response="STOPPED", submit_response="new_job_123")
        job._job_id = "old_job"

        with pytest.raises(TimeoutError):
            await job.stop(timeout_seconds=0)

        run_id = await job.start()
        assert run_id  # new run_id assigned
        assert job.job_id is not None

    @pytest.mark.asyncio
    async def test_reconcile_status_query_failure_preserves_job_id(self) -> None:
        """If the reconciliation status query itself fails, _job_id is preserved
        (conservative — do not clear when we cannot confirm terminal state)."""
        job = _make_job()
        job._job_id = "old_job"
        job._client.get_job_status.side_effect = RuntimeError("network error")

        with pytest.raises(TimeoutError):
            await job.stop(timeout_seconds=0)

        assert job.job_id == "old_job"


class TestGetStatusClearsTerminalJobId:
    @pytest.mark.asyncio
    async def test_get_status_clears_job_id_on_terminal_status(self) -> None:
        """When get_status() observes a terminal state (STOPPED/FAILED/SUCCEEDED),
        _job_id should be cleared so the object doesn't hold a stale reference."""
        for terminal in ("STOPPED", "FAILED", "SUCCEEDED"):
            job = _make_job(get_status_response=terminal)
            job._job_id = "some_job"

            status = await job.get_status()
            assert status in (JobStatus.STOPPED, JobStatus.FAILED)
            assert job.job_id is None, f"job_id not cleared for terminal status {terminal}"

    @pytest.mark.asyncio
    async def test_get_status_preserves_job_id_for_running(self) -> None:
        """Non-terminal status should NOT clear _job_id."""
        job = _make_job(get_status_response="RUNNING")
        job._job_id = "active_job"

        status = await job.get_status()
        assert status == JobStatus.RUNNING
        assert job.job_id == "active_job"
