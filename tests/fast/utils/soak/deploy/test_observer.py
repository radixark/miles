import subprocess
import threading
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import NAMESPACE, ORCHESTRATOR, RELEASE, ROLLOUT_EXECUTOR, TRAINER
from tests.fast.utils.soak.deploy.deploy_fakes import _STATE_FILE, _FakeDeploymentReads, _release_snapshot
from tests.utils.deploy.hot_restart.cluster_observer import ClusterSnapshot, compute_trainer_rpc_url
from tests.utils.soak.deploy.observers import DeploymentObserver
from tests.utils.soak.deploy.types import DeploymentObservationDetails, DeploymentTarget

from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames


def _observer() -> DeploymentObserver:
    return DeploymentObserver(
        namespace=NAMESPACE,
        release=RELEASE,
        trainer_id="trainer-a",
        checkpoint_dir=Path("/dumps/run/checkpoints"),
        events_dir=Path("/dumps/run/events"),
    )


async def _observe(
    monkeypatch: pytest.MonkeyPatch, reads: _FakeDeploymentReads
) -> tuple[list[DeploymentTarget] | None, dict]:
    reads.install(monkeypatch)
    observation = await _observer().observe()
    assert observation.details == DeploymentObservationDetails(cluster=reads.snapshot)
    return observation.targets, observation.errors


class TestDeploymentObserverTarget:
    async def test_a_settled_release_is_one_ready_target_with_its_identity_and_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every field a take-over guards on is copied from the reads of this release."""
        stamps = {ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: "t1", TRAINER: None}

        targets, errors = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(stamps=stamps)))

        [target] = targets
        assert errors == {}
        assert (target.identity, target.release, target.namespace) == (RELEASE, RELEASE, NAMESPACE)
        assert target.alive and target.ready
        assert target.workload_stamps == stamps
        assert target.workload_uids == {name: f"uid-{name}" for name in stamps}
        assert (target.saved_iteration, target.finished_rollout_id) == (3, 5)
        assert target.state_file == _STATE_FILE
        assert target.uninstall_job_uid == "uid-uninstall"

    async def test_the_three_reads_run_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cluster, uninstall job and progress are read in parallel so one observation is one moment."""
        reads = _FakeDeploymentReads(snapshot=_release_snapshot(), barrier=threading.Barrier(3, timeout=10))

        targets, errors = await _observe(monkeypatch, reads)

        assert errors == {}
        assert len(targets) == 1

    async def test_the_reads_address_this_release_and_its_directories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading another release's job, trainer or dump directory would describe a run nobody is taking over."""
        reads = _FakeDeploymentReads(snapshot=_release_snapshot())

        await _observe(monkeypatch, reads)

        assert reads.cluster_kwargs == [
            {
                "release": RELEASE,
                "namespace": NAMESPACE,
                "trainer_rpc_url": compute_trainer_rpc_url(
                    release=RELEASE, namespace=NAMESPACE, trainer_id="trainer-a"
                ),
            }
        ]
        assert reads.progress_kwargs == [
            {"checkpoint_dir": Path("/dumps/run/checkpoints"), "events_dir": Path("/dumps/run/events")}
        ]
        [argv] = reads.job_argv
        assert argv[:4] == ["kubectl", "get", "job", RunNames.uninstall_job(release=RELEASE)]
        assert argv[argv.index("--namespace") + 1] == NAMESPACE
        assert "--ignore-not-found" in argv

    async def test_an_absent_uninstall_job_is_a_ready_target_without_a_job_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--ignore-not-found prints nothing for a missing job, which is an observed absence and not an error."""
        targets, errors = await _observe(
            monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(), job_stdout="")
        )

        assert errors == {}
        assert targets[0].ready and targets[0].uninstall_job_uid is None

    @pytest.mark.parametrize(
        "snapshot",
        [
            pytest.param(_release_snapshot(reads_missing=("statefulsets",)), id="failed_workload_read"),
            pytest.param(_release_snapshot(reads_missing=("pods",)), id="failed_pod_read"),
            pytest.param(_release_snapshot(with_pods=False), id="gone_release"),
            pytest.param(_release_snapshot(stamps={}), id="no_workloads"),
        ],
    )
    async def test_a_failed_partial_or_gone_snapshot_yields_no_target(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: ClusterSnapshot
    ) -> None:
        """Only a whole, present release may be drawn against; anything else would compare against nothing."""
        targets, _ = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=snapshot))

        assert targets == []

    async def test_a_failed_kind_read_is_recorded_as_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing read must stop the scheduler rather than look like a quiet release."""
        _, errors = await _observe(
            monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(reads_missing=("statefulsets",)))
        )

        assert errors == {"statefulsets": "Read failed"}

    async def test_a_missing_hot_restart_workload_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the orchestrator and rollout executor there is nothing a take-over could restamp."""
        snapshot = _release_snapshot(stamps={ORCHESTRATOR: None, TRAINER: None})

        targets, errors = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=snapshot))

        assert errors == {}
        assert not targets[0].ready

    async def test_a_failed_uninstall_job_read_is_an_error_and_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the job identity the guard cannot tell whose uninstall job it would delete."""
        error = subprocess.CalledProcessError(1, ["kubectl"])

        targets, errors = await _observe(
            monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(), job_stdout=error)
        )

        assert set(errors) == {"uninstall_job"}
        assert not targets[0].ready and targets[0].uninstall_job_uid is None

    async def test_a_failed_progress_read_is_an_error_and_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without progress no checkpoint gate can be judged, so the target cannot be drawn."""
        targets, errors = await _observe(
            monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(), progress=OSError("nfs"))
        )

        assert set(errors) == {"progress"}
        assert not targets[0].ready
        assert (targets[0].saved_iteration, targets[0].finished_rollout_id) == (None, None)


class TestDeploymentObserverIncarnation:
    async def test_the_incarnation_ignores_workload_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same stamps listed in another order describe the same generation."""
        forward = {ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: "t1", TRAINER: None}
        backward = dict(reversed(list(forward.items())))

        [first], _ = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(stamps=forward)))
        [second], _ = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(stamps=backward)))

        assert first.incarnation == second.incarnation

    @pytest.mark.parametrize(
        "stamps",
        [
            pytest.param({ORCHESTRATOR: "t2", ROLLOUT_EXECUTOR: "t1", TRAINER: None}, id="one_restamped"),
            pytest.param({ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: None, TRAINER: None}, id="stamp_removed"),
        ],
    )
    async def test_any_stamp_change_is_a_new_incarnation(
        self, monkeypatch: pytest.MonkeyPatch, stamps: dict[str, str | None]
    ) -> None:
        """A take-over is told apart by its stamps, so any rewrite must change the incarnation."""
        base = {ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: "t1", TRAINER: None}

        [before], _ = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(stamps=base)))
        [after], _ = await _observe(monkeypatch, _FakeDeploymentReads(snapshot=_release_snapshot(stamps=stamps)))

        assert before.incarnation != after.incarnation
