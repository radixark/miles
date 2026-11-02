import pytest

from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)
from tests.utils.deploy.hot_restart.cluster_observer import SnapshotRecorder
from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence
from tests.utils.soak.deploy.checkers.takeover_scope import assert_take_overs_carried_rollout_only_args
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.serving.worker_config import ServeWorkerConfig


class TestTakeoverPayloadScope:
    def test_repeated_observations_do_not_add_argument_generations(self) -> None:
        """A pod observed repeatedly counts once while each replacement carries its own value."""
        assert_take_overs_carried_rollout_only_args(
            _evidence(), flag="--save-debug-rollout-data", values=["/gen0", "/gen1"]
        )

    @pytest.mark.parametrize(
        "failure,message",
        [
            ("missing", "no read"),
            ("wrong", "arguments other"),
            ("leaked", "argument leaked"),
            ("no_config", "carries none"),
            ("missing_generation", "cannot be paired"),
        ],
    )
    def test_missing_reordered_or_leaked_payloads_fail(self, failure: str, message: str) -> None:
        """The checker rejects missing commands, wrong generations and rollout arguments in trainer payloads."""
        evidence = _evidence()
        commands = dict(evidence.commands_of_pod_uid)
        if failure == "missing":
            del commands["r1"]
        elif failure == "wrong":
            commands["r1"] = commands["r0"]
        elif failure == "leaked":
            commands["t"] = ("python", "/gen1")
        elif failure == "no_config":
            commands["r1"] = ("python",)
        elif failure == "missing_generation":
            evidence = evidence.model_copy(update={"snapshots": evidence.snapshots[:1]})
        evidence = evidence.model_copy(update={"commands_of_pod_uid": commands})
        with pytest.raises(AssertionError, match=message):
            assert_take_overs_carried_rollout_only_args(
                evidence, flag="--save-debug-rollout-data", values=["/gen0", "/gen1"]
            )


class TestObservedCommands:
    def test_replacement_keeps_the_old_command_without_duplicating_it_in_each_snapshot(self) -> None:
        """Command evidence survives replacement and is stored once for every observed pod UID."""
        evidence = _evidence()
        recorder = SnapshotRecorder(release=RELEASE)
        for snapshot in evidence.snapshots:
            recorder.record(
                snapshot.model_copy(
                    update={"commands_of_pod_uid": {p.uid: evidence.commands_of_pod_uid[p.uid] for p in snapshot.pods}}
                )
            )
        assert recorder.commands_of_pod_uid == evidence.commands_of_pod_uid
        assert recorder.snapshots
        assert all(snapshot.commands_of_pod_uid == {} for snapshot in recorder.snapshots)


def _evidence() -> HotRestartEvidence:
    snapshots = tuple(
        cluster_snapshot(
            pods=[pod_fact(f"{ROLLOUT_EXECUTOR}-0", uid=uid), pod_fact(f"{TRAINER}-0", uid="t")],
            workloads=[workload_fact(ROLLOUT_EXECUTOR), workload_fact(TRAINER)],
        )
        for uid in ("r0", "r0", "r1")
    )
    commands = {"t": ("python", "trainer")}
    for index in range(2):
        payload = ServeWorkerConfig(
            worker_type="rollout_executor",
            args={"save_debug_rollout_data": f"/gen{index}"},
            static_connections=StaticConnConfig(),
        )
        commands[f"r{index}"] = ("python", "--config", payload.model_dump_json())
    return HotRestartEvidence(records=(), snapshots=snapshots, release=RELEASE, commands_of_pod_uid=commands)
