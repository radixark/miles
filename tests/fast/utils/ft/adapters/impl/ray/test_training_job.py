from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob, _parse_ray_status, _stop_job, stop_all_active_jobs
from miles.utils.ft.adapters.types import JobStatus


def _make_job(
    entrypoint: str = "python train.py",
    runtime_env: dict[str, Any] | None = None,
    poll_interval_seconds: float = 0,
    ft_id: str = "test",
    k8s_label_prefix: str = "",
) -> tuple[RayMainJob, MagicMock]:
    mock_client = MagicMock()
    job = RayMainJob(
        client=mock_client,
        entrypoint=entrypoint,
        ft_id=ft_id,
        k8s_label_prefix=k8s_label_prefix,
        runtime_env=runtime_env,
        poll_interval_seconds=poll_interval_seconds,
    )
    return job, mock_client


class TestSubmitJob:
    @pytest.mark.anyio
    async def test_calls_submit_job_and_returns_run_id(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "ray-job-abc"

        run_id = await job.start()

        assert isinstance(run_id, str)
        assert len(run_id) == 8
        mock_client.submit_job.assert_called_once()
        call_kwargs = mock_client.submit_job.call_args.kwargs
        assert call_kwargs["entrypoint"] == "python train.py"
        assert call_kwargs["runtime_env"]["env_vars"]["MILES_FT_TRAINING_RUN_ID"] == run_id

    @pytest.mark.anyio
    async def test_submit_raises_when_previous_job_is_active(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        with pytest.raises(RuntimeError, match="previous job.*still tracked"):
            await job.start()

    @pytest.mark.anyio
    async def test_submit_succeeds_after_previous_job_stopped(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.side_effect = ["job-1", "job-2"]
        mock_client.get_job_status.return_value = "STOPPED"

        run_id_1 = await job.start()
        await job.stop(timeout_seconds=10)
        run_id_2 = await job.start()

        assert run_id_1 != run_id_2
        assert mock_client.submit_job.call_count == 2

    @pytest.mark.anyio
    async def test_injects_ft_id_and_label_prefix_into_env(self) -> None:
        mock_client = MagicMock()
        job = RayMainJob(
            client=mock_client,
            entrypoint="python train.py",
            ft_id="myft",
            k8s_label_prefix="pfx1",
        )
        mock_client.submit_job.return_value = "ray-job-ft"

        await job.start()

        call_kwargs = mock_client.submit_job.call_args.kwargs
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert env_vars["MILES_FT_ID"] == "myft"
        assert env_vars["MILES_FT_K8S_LABEL_PREFIX"] == "pfx1"

    @pytest.mark.anyio
    async def test_job_id_none_before_submit_and_set_after(self) -> None:
        job, mock_client = _make_job()
        assert job.job_id is None

        mock_client.submit_job.return_value = "ray-job-123"
        await job.start()
        assert job.job_id == "ray-job-123"

    @pytest.mark.anyio
    async def test_does_not_mutate_original_runtime_env(self) -> None:
        original_env: dict[str, Any] = {"working_dir": "/data", "env_vars": {"MY_VAR": "1"}}
        original_env_copy = {
            "working_dir": "/data",
            "env_vars": {"MY_VAR": "1"},
        }
        job, mock_client = _make_job(runtime_env=original_env)
        mock_client.submit_job.return_value = "ray-job-xyz"

        await job.start()

        assert original_env == original_env_copy


class TestStopJob:
    @pytest.mark.anyio
    async def test_stop_polls_until_stopped(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.side_effect = [
            "RUNNING",
            "RUNNING",
            "STOPPED",
        ]

        await job.stop(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")
        assert mock_client.get_job_status.call_count == 3

    @pytest.mark.anyio
    async def test_stop_raises_timeout_error(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.return_value = "RUNNING"

        with patch("miles.utils.ft.adapters.impl.ray.main_job.time") as mock_time, patch(
            "miles.utils.ft.utils.polling.time"
        ) as mock_poll_time:
            call_count = 0

            def advancing_monotonic() -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count * 100)

            mock_time.monotonic = advancing_monotonic
            mock_poll_time.monotonic = advancing_monotonic

            with pytest.raises(TimeoutError, match="stop_job"):
                await job.stop(timeout_seconds=5)

    @pytest.mark.anyio
    async def test_stop_completes_when_job_fails(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.side_effect = ["RUNNING", "FAILED"]

        await job.stop(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")
        assert mock_client.get_job_status.call_count == 2

    @pytest.mark.anyio
    async def test_stop_completes_when_job_succeeds(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.return_value = "SUCCEEDED"

        await job.stop(timeout_seconds=10)

        mock_client.stop_job.assert_called_once_with("job-1")

    @pytest.mark.anyio
    async def test_stop_with_no_active_job_is_noop(self) -> None:
        job, mock_client = _make_job()

        await job.stop()

        mock_client.stop_job.assert_not_called()

    @pytest.mark.anyio
    async def test_get_status_returns_stopped_after_successful_stop(self) -> None:
        """After stop_job, get_job_status returns STOPPED without calling Ray."""
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.return_value = "STOPPED"
        await job.stop(timeout_seconds=10)

        mock_client.get_job_status.reset_mock()
        status = await job.get_status()
        assert status == JobStatus.STOPPED
        mock_client.get_job_status.assert_not_called()

    @pytest.mark.anyio
    async def test_job_id_cleared_after_stop(self) -> None:
        job, mock_client = _make_job(poll_interval_seconds=0)
        mock_client.submit_job.return_value = "job-1"
        await job.start()
        assert job.job_id == "job-1"

        mock_client.get_job_status.return_value = "STOPPED"
        await job.stop(timeout_seconds=10)
        assert job.job_id is None


class TestGetJobStatus:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("raw_status", "expected"),
        [
            ("RUNNING", JobStatus.RUNNING),
            ("FAILED", JobStatus.FAILED),
            ("SUCCEEDED", JobStatus.STOPPED),
            ("PENDING", JobStatus.PENDING),
            ("UNKNOWN_STATE", JobStatus.FAILED),
        ],
    )
    async def test_status_mapping(
        self,
        raw_status: str,
        expected: JobStatus,
    ) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.return_value = raw_status
        status = await job.get_status()
        assert status == expected

    @pytest.mark.anyio
    async def test_no_job_returns_stopped(self) -> None:
        job, _mock_client = _make_job()

        status = await job.get_status()
        assert status == JobStatus.STOPPED

    @pytest.mark.anyio
    async def test_raises_on_client_failure(self) -> None:
        job, mock_client = _make_job()
        mock_client.submit_job.return_value = "job-1"
        await job.start()

        mock_client.get_job_status.side_effect = ConnectionError("Ray unreachable")

        with pytest.raises(ConnectionError, match="Ray unreachable"):
            await job.get_status()


class TestParseRayStatus:
    def test_plain_string(self) -> None:
        assert _parse_ray_status("RUNNING") == "RUNNING"

    def test_enum_style_dotted_string(self) -> None:
        assert _parse_ray_status("JobStatus.RUNNING") == "RUNNING"

    def test_deeply_dotted(self) -> None:
        assert _parse_ray_status("ray.job_submission.JobSubmissionStatus.FAILED") == "FAILED"

    def test_mock_enum_object(self) -> None:
        class FakeEnum:
            def __str__(self) -> str:
                return "JobSubmissionStatus.SUCCEEDED"

        assert _parse_ray_status(FakeEnum()) == "SUCCEEDED"


class TestStopJob:
    @pytest.mark.anyio
    async def test_polls_until_terminal(self) -> None:
        mock_client = MagicMock()
        mock_client.get_job_status.side_effect = ["RUNNING", "RUNNING", "STOPPED"]

        await _stop_job(client=mock_client, job_id="job-1", timeout_seconds=10, poll_interval=0)

        mock_client.stop_job.assert_called_once_with("job-1")
        assert mock_client.get_job_status.call_count == 3

    @pytest.mark.anyio
    async def test_raises_timeout_when_job_stays_active(self) -> None:
        mock_client = MagicMock()
        mock_client.get_job_status.return_value = "RUNNING"

        with patch("miles.utils.ft.adapters.impl.ray.main_job.time") as mock_time, patch(
            "miles.utils.ft.utils.polling.time"
        ) as mock_poll_time:
            call_count = 0

            def advancing_monotonic() -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count * 100)

            mock_time.monotonic = advancing_monotonic
            mock_poll_time.monotonic = advancing_monotonic

            with pytest.raises(TimeoutError, match="stop_job"):
                await _stop_job(client=mock_client, job_id="job-1", timeout_seconds=5, poll_interval=0)

    @pytest.mark.anyio
    async def test_raises_immediately_when_stop_rpc_exhausts_timeout_budget(self) -> None:
        """When stop_job RPC takes longer than the overall timeout, raise immediately with a clear message."""
        mock_client = MagicMock()

        with patch("miles.utils.ft.adapters.impl.ray.main_job.time") as mock_time:
            timestamps = iter([0.0, 15.0])
            mock_time.monotonic = lambda: next(timestamps)

            with pytest.raises(TimeoutError, match="no time left for polling"):
                await _stop_job(client=mock_client, job_id="job-1", timeout_seconds=10, poll_interval=0)

    @pytest.mark.anyio
    async def test_accepts_failed_as_terminal(self) -> None:
        mock_client = MagicMock()
        mock_client.get_job_status.return_value = "FAILED"

        await _stop_job(client=mock_client, job_id="job-1", timeout_seconds=10, poll_interval=0)

        mock_client.stop_job.assert_called_once_with("job-1")

    @pytest.mark.anyio
    async def test_accepts_succeeded_as_terminal(self) -> None:
        mock_client = MagicMock()
        mock_client.get_job_status.return_value = "SUCCEEDED"

        await _stop_job(client=mock_client, job_id="job-1", timeout_seconds=10, poll_interval=0)

        mock_client.stop_job.assert_called_once_with("job-1")


class _FakeJobDetails:
    """Minimal stand-in for ray.job_submission.JobDetails."""

    def __init__(self, job_id: str, status: str) -> None:
        self.job_id = job_id
        self.submission_id = job_id
        self.status = status


class TestStopAllActiveJobs:
    @pytest.mark.anyio
    async def test_stops_all_active_jobs(self) -> None:
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            _FakeJobDetails("job-a", "RUNNING"),
            _FakeJobDetails("job-b", "PENDING"),
            _FakeJobDetails("job-c", "STOPPED"),
        ]
        mock_client.get_job_status.return_value = "STOPPED"

        count = await stop_all_active_jobs(client=mock_client, timeout_seconds=10)

        assert count == 2
        stopped_ids = [call.args[0] for call in mock_client.stop_job.call_args_list]
        assert "job-a" in stopped_ids
        assert "job-b" in stopped_ids
        assert "job-c" not in stopped_ids

    @pytest.mark.anyio
    async def test_returns_zero_when_no_active_jobs(self) -> None:
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            _FakeJobDetails("job-x", "STOPPED"),
            _FakeJobDetails("job-y", "FAILED"),
        ]

        count = await stop_all_active_jobs(client=mock_client, timeout_seconds=10)

        assert count == 0
        mock_client.stop_job.assert_not_called()

    @pytest.mark.anyio
    async def test_continues_on_individual_stop_failure(self) -> None:
        """If stopping one job fails, other jobs should still be attempted."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            _FakeJobDetails("job-ok", "RUNNING"),
            _FakeJobDetails("job-fail", "RUNNING"),
        ]

        def side_effect_stop(job_id: str) -> None:
            if job_id == "job-fail":
                raise RuntimeError("connection lost")

        mock_client.stop_job.side_effect = side_effect_stop
        mock_client.get_job_status.return_value = "STOPPED"

        count = await stop_all_active_jobs(client=mock_client, timeout_seconds=10)

        assert count == 1
        assert mock_client.stop_job.call_count == 2

    @pytest.mark.anyio
    async def test_stop_all_returns_exact_success_count(self) -> None:
        """With 3 active jobs and 1 failure, returns 2."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            _FakeJobDetails("job-a", "RUNNING"),
            _FakeJobDetails("job-b", "RUNNING"),
            _FakeJobDetails("job-c", "RUNNING"),
        ]

        def side_effect_stop(job_id: str) -> None:
            if job_id == "job-b":
                raise RuntimeError("timeout")

        mock_client.stop_job.side_effect = side_effect_stop
        mock_client.get_job_status.return_value = "STOPPED"

        count = await stop_all_active_jobs(client=mock_client, timeout_seconds=10)

        assert count == 2
        assert mock_client.stop_job.call_count == 3
