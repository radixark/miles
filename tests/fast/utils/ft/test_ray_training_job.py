from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.platform.protocols import JobStatus
from miles.utils.ft.platform.ray_training_job import RayTrainingJob


def _make_job(
    entrypoint: str = "python train.py",
    runtime_env: dict[str, Any] | None = None,
    poll_interval_seconds: int = 0,
) -> tuple[RayTrainingJob, MagicMock]:
    mock_client = MagicMock()
    job = RayTrainingJob(
        client=mock_client,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        poll_interval_seconds=poll_interval_seconds,
    )
    return job, mock_client


class TestSubmitTraining:
    @pytest.mark.asyncio
    async def test_calls_submit_job_and_returns_run_id(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "ray-job-abc"

        run_id = await job.submit_training()

        assert isinstance(run_id, str)
        assert len(run_id) == 8
        mock_client.submit_job.assert_called_once()
        call_kwargs = mock_client.submit_job.call_args.kwargs
        assert call_kwargs["entrypoint"] == "python train.py"
        assert call_kwargs["runtime_env"]["env_vars"]["FT_TRAINING_RUN_ID"] == run_id

    @pytest.mark.asyncio
    async def test_multiple_submits_produce_different_run_ids(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.side_effect = ["job-1", "job-2"]

        run_id_1 = await job.submit_training()
        run_id_2 = await job.submit_training()

        assert run_id_1 != run_id_2
        assert mock_client.submit_job.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_mutate_original_runtime_env(self) -> None:
        original_env: dict[str, Any] = {"working_dir": "/data", "env_vars": {"MY_VAR": "1"}}
        original_env_copy = {
            "working_dir": "/data",
            "env_vars": {"MY_VAR": "1"},
        }
        job, mock_client = _make_job(runtime_env=original_env)
        mock_client.submit_job.return_value = "ray-job-xyz"

        await job.submit_training()

        assert original_env == original_env_copy


class TestStopTraining:
    @pytest.mark.asyncio
    async def test_stop_polls_until_stopped(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.side_effect = [
            "RUNNING",
            "RUNNING",
            "STOPPED",
        ]

        await job.stop_training(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")
        assert mock_client.get_job_status.call_count == 3

    @pytest.mark.asyncio
    async def test_stop_raises_timeout_error(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "RUNNING"

        with patch("miles.utils.ft.platform.ray_training_job.time") as mock_time:
            call_count = 0

            def advancing_monotonic() -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count * 100)

            mock_time.monotonic = advancing_monotonic

            with pytest.raises(TimeoutError, match="did not stop within"):
                await job.stop_training(timeout_seconds=5)

    @pytest.mark.asyncio
    async def test_stop_completes_when_job_fails(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.side_effect = ["RUNNING", "FAILED"]

        await job.stop_training(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")
        assert mock_client.get_job_status.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_completes_when_job_succeeds(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "SUCCEEDED"

        await job.stop_training(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")

    @pytest.mark.asyncio
    async def test_stop_with_no_active_job_is_noop(self) -> None:
        job, mock_client = _make_job()

        await job.stop_training()

        mock_client.stop_job.assert_not_called()


class TestGetTrainingStatus:
    @pytest.mark.asyncio
    async def test_running_status(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "RUNNING"
        status = await job.get_training_status()
        assert status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_failed_status(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "FAILED"
        status = await job.get_training_status()
        assert status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_succeeded_maps_to_stopped(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "SUCCEEDED"
        status = await job.get_training_status()
        assert status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_no_job_returns_stopped(self) -> None:
        job, _mock_client = _make_job()

        status = await job.get_training_status()
        assert status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pending_status(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "PENDING"
        status = await job.get_training_status()
        assert status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_unknown_status_maps_to_failed(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.return_value = "UNKNOWN_STATE"
        status = await job.get_training_status()
        assert status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_raises_on_client_failure(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.submit_training()

        mock_client.get_job_status.side_effect = ConnectionError("Ray unreachable")

        with pytest.raises(ConnectionError, match="Ray unreachable"):
            await job.get_training_status()
