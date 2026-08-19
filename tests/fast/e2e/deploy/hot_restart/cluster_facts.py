from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    STATEFUL_SET_KIND,
    ClusterSnapshot,
    PodFact,
    WorkloadFact,
)

NAMESPACE: str = "rl"
RELEASE: str = "miles-run-demo-all"
ORCHESTRATOR: str = "miles-run-demo-all-orchestrator"
ROLLOUT_EXECUTOR: str = "miles-run-demo-all-rollout-executor"
TRAINER: str = "miles-run-demo-all-trainer-controller-actor"
ENGINE_POOL: str = "miles-run-demo-all-rollout-engine"


def cluster_snapshot(
    *,
    pods: list[PodFact],
    workloads: list[WorkloadFact],
    trainer_boot_uuid: str | None = "boot-a",
    reads_missing: tuple[str, ...] = (),
) -> ClusterSnapshot:
    return ClusterSnapshot(
        pods=tuple(pods),
        workloads=tuple(workloads),
        trainer_boot_uuid=trainer_boot_uuid,
        reads_missing=reads_missing,
    )


def pod_fact(name: str, *, uid: str, restart_count: int = 0) -> PodFact:
    return PodFact(name=name, uid=uid, restart_count=restart_count)


def workload_fact(
    name: str,
    *,
    kind: str = STATEFUL_SET_KIND,
    generation: int = 1,
    pod_template_fingerprint: str = "template-a",
    restart_at: str | None = None,
) -> WorkloadFact:
    return WorkloadFact(
        kind=kind,
        name=name,
        generation=generation,
        pod_template_fingerprint=pod_template_fingerprint,
        restart_at=restart_at,
    )
