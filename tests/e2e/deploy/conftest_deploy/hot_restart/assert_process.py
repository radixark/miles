from collections.abc import Sequence

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterSnapshot
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence

MINIMUM_COMPLETE_SNAPSHOTS: int = 2
MINIMUM_OBSERVATION_SUCCESS_RATIO: float = 0.5


def assert_run_watched_closely(evidence: HotRestartEvidence) -> None:
    complete = [one for one in evidence.snapshots if one.describes_whole_release]

    assert len(complete) >= MINIMUM_COMPLETE_SNAPSHOTS, (
        f"the whole release was read {len(complete)} time(s) out of {len(evidence.snapshots)} observation(s), fewer "
        f"than the {MINIMUM_COMPLETE_SNAPSHOTS} it takes to tell a pod that survived every restart from one that "
        f"was never looked at twice"
    )

    if (attempts := evidence.observation_attempts) > 0:
        assert len(complete) >= MINIMUM_OBSERVATION_SUCCESS_RATIO * attempts, (
            f"the whole release was read {len(complete)} time(s) out of {attempts} attempt(s), under "
            f"{MINIMUM_OBSERVATION_SUCCESS_RATIO:.0%}; a cluster that answers half the time hides the very pod "
            f"replacement these assertions are here to catch"
        )


def assert_trainer_not_rebooted(evidence: HotRestartEvidence) -> None:
    boot_uuids = _compute_trainer_boot_uuids(evidence.snapshots)
    assert boot_uuids, (
        f"the trainer's rpc server was never reached across {len(evidence.snapshots)} observation(s), so nothing "
        f"here says whether the process a take-over reloaded a checkpoint into is the one that had been training"
    )
    assert len(boot_uuids) == 1, (
        f"the trainer's rpc server answered with the boot uuid(s) {sorted(boot_uuids)}, not one: the process a "
        f"take-over reloaded a checkpoint into is not the process that had been training"
    )

    for stamp, index in sorted(_compute_first_snapshot_index_of_stamp(evidence.snapshots).items()):
        answered = [one for one in evidence.snapshots[index:] if one.trainer_boot_uuid is not None]
        assert answered, (
            f"the take-over stamped {stamp} and the trainer's rpc server was never reached after it, so the one "
            f"uuid this run collected only proves the trainer was alive before its script was replaced"
        )


def _compute_trainer_boot_uuids(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    return {one.trainer_boot_uuid for one in snapshots if one.trainer_boot_uuid is not None}


def _compute_first_snapshot_index_of_stamp(snapshots: Sequence[ClusterSnapshot]) -> dict[str, int]:
    first_index_of_stamp: dict[str, int] = {}
    for index, snapshot in enumerate(snapshots):
        for workload in snapshot.workloads:
            if workload.restart_at is not None:
                first_index_of_stamp.setdefault(workload.restart_at, index)
    return first_index_of_stamp
