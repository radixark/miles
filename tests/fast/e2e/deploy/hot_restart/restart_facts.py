from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    STATEFUL_SET_KIND,
    ClusterSnapshot,
    PodFact,
    WorkloadFact,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    ORCHESTRATOR,
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
)

ENGINE_POOL_POD: str = f"{ENGINE_POOL}-0-1"


def restart_snapshot(
    *, uid_of_pod: dict[str, str], stamp_of_workload: dict[str, str | None], boot_uuid: str | None = "boot-a"
) -> ClusterSnapshot:
    return ClusterSnapshot(
        pods=tuple(PodFact(name=name, uid=uid, restart_count=0) for name, uid in sorted(uid_of_pod.items())),
        workloads=tuple(
            WorkloadFact(
                kind=STATEFUL_SET_KIND,
                name=name,
                generation=1 if stamp is None else 2,
                pod_template_fingerprint="template-base" if stamp is None else f"template-{stamp}",
                restart_at=stamp,
            )
            for name, stamp in sorted(stamp_of_workload.items())
        ),
        trainer_boot_uuid=boot_uuid,
    )


def evidence_of(
    *,
    snapshots: list[ClusterSnapshot],
    records: list[HotRestartRecord],
    observation_attempts: int | None = None,
    observation_failures: int = 0,
) -> HotRestartEvidence:
    return HotRestartEvidence(
        records=tuple(records),
        snapshots=tuple(snapshots),
        release=RELEASE,
        observation_attempts=len(snapshots) if observation_attempts is None else observation_attempts,
        observation_failures=observation_failures,
    )


def quiet_run_snapshot(*, uid: str = "uid-o-1") -> ClusterSnapshot:
    return restart_snapshot(
        uid_of_pod={f"{ORCHESTRATOR}-0": uid, f"{ROLLOUT_EXECUTOR}-0": "uid-r-1", f"{TRAINER}-0": "uid-t"},
        stamp_of_workload={ORCHESTRATOR: None, ROLLOUT_EXECUTOR: None, TRAINER: None},
    )


def restarted_snapshot(*, stamp: str, uid_suffix: str, boot_uuid: str | None = "boot-a") -> ClusterSnapshot:
    return restart_snapshot(
        uid_of_pod={
            f"{ORCHESTRATOR}-0": f"uid-o-{uid_suffix}",
            f"{ROLLOUT_EXECUTOR}-0": f"uid-r-{uid_suffix}",
            f"{TRAINER}-0": "uid-t",
        },
        stamp_of_workload={ORCHESTRATOR: stamp, ROLLOUT_EXECUTOR: stamp, TRAINER: None},
        boot_uuid=boot_uuid,
    )


def two_restarts() -> list[ClusterSnapshot]:
    return [
        quiet_run_snapshot(),
        restarted_snapshot(stamp="t1", uid_suffix="2"),
        restarted_snapshot(stamp="t2", uid_suffix="3"),
    ]
