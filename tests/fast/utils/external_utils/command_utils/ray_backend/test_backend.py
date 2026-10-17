from typing import Literal

import pytest
from tests.fast.utils.external_utils.command_utils.fake_launch_guard import RecordingLaunchGuard
from tests.fast.utils.external_utils.command_utils.ray_backend.conftest import RecordedRayLaunch

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, LaunchGuard

_STOP_THE_LAUNCHER_CLUSTER = "ray stop --force"


class TestTheSubmissionIdOfARayLaunch:
    @pytest.mark.parametrize("job_lifetime", ["independent", "launcher"])
    @pytest.mark.parametrize("submission_id", [None, "miles-job-7"])
    def test_the_configured_submission_id_and_lifetime_reach_the_job_submission(
        self,
        recorded_ray_launch: RecordedRayLaunch,
        job_lifetime: Literal["independent", "launcher"],
        submission_id: str | None,
    ) -> None:
        """Only a submission id that reaches ray job submit lets the launcher name the job it started."""
        _launch(job_lifetime=job_lifetime, submission_id=submission_id)

        assert len(recorded_ray_launch.submitted) == 1
        submitted = recorded_ray_launch.submitted[0]
        assert submitted["submission_id"] == submission_id
        assert submitted["job_lifetime"] == job_lifetime


class TestTheLauncherOwnedRayCluster:
    def test_a_launcher_owned_job_stops_the_cluster_it_started_after_the_job(
        self, recorded_ray_launch: RecordedRayLaunch
    ) -> None:
        """A launcher-owned job must not leave the ray head it started running after the launcher returns."""
        _launch(job_lifetime="launcher", submission_id="miles-job-7")

        assert recorded_ray_launch.cpu_commands[-1] == _STOP_THE_LAUNCHER_CLUSTER

    def test_a_launcher_owned_job_that_fails_still_stops_the_cluster_and_raises(
        self, recorded_ray_launch: RecordedRayLaunch
    ) -> None:
        """The cleanup sits in a finally, so a failed job must neither leak the head nor hide its failure."""
        recorded_ray_launch.submit_error = RuntimeError("Ray job miles-job-7 ended with FAILED")

        with pytest.raises(RuntimeError, match="miles-job-7"):
            _launch(job_lifetime="launcher", submission_id="miles-job-7")

        assert recorded_ray_launch.cpu_commands[-1] == _STOP_THE_LAUNCHER_CLUSTER

    def test_an_independent_job_keeps_the_cluster_it_runs_on(self, recorded_ray_launch: RecordedRayLaunch) -> None:
        """An independent job outlives the launcher, and stopping ray under it would kill the job."""
        _launch(job_lifetime="independent", submission_id="miles-job-7")

        assert _STOP_THE_LAUNCHER_CLUSTER not in recorded_ray_launch.cpu_commands

    def test_an_external_cluster_is_neither_started_nor_stopped(
        self, recorded_ray_launch: RecordedRayLaunch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher does not own an external ray cluster, so it must leave it running for others."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        _launch(job_lifetime="launcher", submission_id="miles-job-7")

        assert _STOP_THE_LAUNCHER_CLUSTER not in recorded_ray_launch.cpu_commands
        assert not any("ray start" in command for command in recorded_ray_launch.cpu_commands)
        assert len(recorded_ray_launch.submitted) == 1


class TestARayLaunchWithAGuard:
    def test_a_ray_launch_submits_without_consulting_a_guard(self, recorded_ray_launch: RecordedRayLaunch) -> None:
        """A ray launch installs no release, so the guard hooks of a helm install have nothing to guard here."""
        guard = RecordingLaunchGuard()

        _launch(job_lifetime="independent", submission_id=None, guard=guard)

        assert guard.calls == []
        assert len(recorded_ray_launch.submitted) == 1

    def test_extra_manifests_are_refused_before_any_command_runs(self, recorded_ray_launch: RecordedRayLaunch) -> None:
        """A ray launch cannot install manifests, and refusing after cleanup would already have killed a run."""
        with pytest.raises(AssertionError, match="extra_manifests"):
            ExecuteTrainConfig().create_backend().execute_train(
                train_args="--train-backend fsdp",
                num_gpus_per_node=8,
                megatron_model_type=None,
                extra_manifests=["kind: ConfigMap"],
            )

        assert recorded_ray_launch.cpu_commands == []
        assert recorded_ray_launch.submitted == []


def _launch(
    *,
    job_lifetime: Literal["independent", "launcher"],
    submission_id: str | None,
    guard: LaunchGuard | None = None,
) -> None:
    ExecuteTrainConfig(ray_submission_id=submission_id).create_backend().execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=8,
        megatron_model_type=None,
        job_lifetime=job_lifetime,
        guard=guard,
    )
