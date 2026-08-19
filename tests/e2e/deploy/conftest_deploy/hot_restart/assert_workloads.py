from collections.abc import Iterable, Sequence

from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    assert_run_watched_closely,
    assert_trainer_not_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    ClusterSnapshot,
    compute_hot_restart_workloads,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence


# ============================ what a take-over rolls ==========================


def assert_take_overs_replaced_only_script(
    evidence: HotRestartEvidence, *, num_restarts: int, minimum_restarts: int
) -> None:
    assert num_restarts >= minimum_restarts, (
        f"the run was taken over {num_restarts} time(s) of the {minimum_restarts} it needs, so it ran to the end "
        f"without its orchestration script ever being replaced"
    )

    assert_run_watched_closely(evidence)
    assert_only_orchestration_restarted(evidence, num_restarts=num_restarts)
    assert_trainer_not_rebooted(evidence)


def assert_only_orchestration_restarted(evidence: HotRestartEvidence, *, num_restarts: int) -> None:
    expected = compute_hot_restart_workloads(evidence.release)

    unattributed = _compute_unattributed_pod_names(evidence.snapshots)
    assert not unattributed, (
        f"the pods {sorted(unattributed)} belong to no workload this run listed, so nothing would notice them "
        f"being replaced; every pod has to be owned by a statefulset or leaderworkerset of the release"
    )

    replaced = _compute_workloads_with_replaced_pods(evidence.snapshots)
    assert set(replaced) == expected, (
        f"a hot restart replaces the pods of {sorted(expected)} and leaves every other pod running; these lost a "
        f"pod instead: {replaced}"
    )

    rolled = _compute_workloads_with_changed_template(evidence.snapshots)
    assert rolled == expected, (
        f"only {sorted(expected)} may be rolled by a hot restart, and the pod template of {sorted(rolled)} "
        f"changed: the relaunch rewrote the run's trainers or engines"
    )

    stamps_of_workload = _compute_restart_stamps_of_workload(evidence.snapshots)
    for workload in sorted(expected):
        assert len(stamps := stamps_of_workload[workload]) == num_restarts, (
            f"{workload} was observed carrying {sorted(stamps)}, and {num_restarts} hot restart(s) stamp one value "
            f"each, so a restart either never reached this workload or never landed"
        )
    unexpected = {
        name: sorted(stamps) for name, stamps in stamps_of_workload.items() if stamps and name not in expected
    }
    assert (
        not unexpected
    ), f"a hot restart stamps exactly the two pod templates it replaces, and these carry a stamp too: {unexpected}"


# ========================== what the snapshots say ============================


def _compute_unattributed_pod_names(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    return {
        pod.name
        for snapshot in snapshots
        if snapshot.describes_whole_release
        for pod in snapshot.pods
        if _compute_workload_of_pod(pod.name, workloads=snapshot.workload_names) is None
    }


def _compute_workloads_with_replaced_pods(snapshots: Sequence[ClusterSnapshot]) -> dict[str, list[str]]:
    workloads = sorted({name for snapshot in snapshots for name in snapshot.workload_names})
    uids_of_pod: dict[str, set[str]] = {}
    restart_counts_of_pod: dict[str, set[int]] = {}
    pod_names_of_workload: dict[str, set[frozenset[str]]] = {}

    for snapshot in snapshots:
        seen_of_workload: dict[str, set[str]] = {one: set() for one in workloads}
        for pod in snapshot.pods:
            if (workload := _compute_workload_of_pod(pod.name, workloads=workloads)) is None:
                continue
            seen_of_workload[workload].add(pod.name)
            uids_of_pod.setdefault(pod.name, set()).add(pod.uid)
            restart_counts_of_pod.setdefault(pod.name, set()).add(pod.restart_count)
        for workload, seen in seen_of_workload.items():
            pod_names_of_workload.setdefault(workload, set()).add(frozenset(seen))

    reasons_of_workload: dict[str, list[str]] = {}
    for pod_name in sorted(uids_of_pod):
        workload = _compute_workload_of_pod(pod_name, workloads=workloads)
        assert workload is not None
        if len(uids := uids_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(f"pod {pod_name} was recreated as {sorted(uids)}")
        if len(counts := restart_counts_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"pod {pod_name} restarted a container: restartCount went through {sorted(counts)}"
            )
    for workload in workloads:
        if len(name_sets := pod_names_of_workload.get(workload, set())) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"the pods of {workload} came and went: {sorted(sorted(one) for one in name_sets)}"
            )

    return {workload: sorted(reasons) for workload, reasons in sorted(reasons_of_workload.items())}


def _compute_workloads_with_changed_template(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    fingerprints_of_workload: dict[str, set[str]] = {}
    stamps_of_workload = _compute_restart_stamps_of_workload(snapshots)
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            fingerprints_of_workload.setdefault(workload.name, set()).add(workload.pod_template_fingerprint)
    return {
        name
        for name, fingerprints in fingerprints_of_workload.items()
        if len(fingerprints) > 1 or len(stamps_of_workload.get(name, set())) > 1
    }


def _compute_restart_stamps_of_workload(snapshots: Sequence[ClusterSnapshot]) -> dict[str, set[str]]:
    stamps_of_workload: dict[str, set[str]] = {}
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            stamps = stamps_of_workload.setdefault(workload.name, set())
            if workload.restart_at is not None:
                stamps.add(workload.restart_at)
    return stamps_of_workload


def _compute_workload_of_pod(pod_name: str, *, workloads: Iterable[str]) -> str | None:
    candidates = [one for one in workloads if pod_name.startswith(f"{one}-")]
    return max(candidates, key=len) if candidates else None
