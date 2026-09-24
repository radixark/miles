from collections.abc import Iterable, Sequence

from tests.utils.deploy.hot_restart.assert_process import (
    assert_baseline_read_before_first_take_over,
    assert_run_watched_closely,
    assert_trainer_not_rebooted,
)
from tests.utils.deploy.hot_restart.cluster_observer import ClusterSnapshot, compute_hot_restart_workloads
from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT
from miles.utils.workers.serving.utils import parse_serve_worker_config
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

SERVE_CONFIG_FLAG: str = "--config"


# ============================ what a take-over rolls ==========================


def assert_take_overs_replaced_only_script(
    evidence: HotRestartEvidence, *, num_restarts: int, minimum_restarts: int
) -> None:
    assert num_restarts >= minimum_restarts, (
        f"the run was taken over {num_restarts} time(s) of the {minimum_restarts} it needs, so it ran to the end "
        f"without its orchestration script ever being replaced"
    )

    assert_run_watched_closely(evidence)
    assert_baseline_read_before_first_take_over(evidence)
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


# ======================== what a take-over's pods carry =======================


def assert_take_overs_carried_rollout_only_args(
    evidence: HotRestartEvidence, *, flag: str, values: Sequence[str]
) -> None:
    orchestrator = component_name(evidence.release, ORCHESTRATOR_COMPONENT)
    rollout_executor = component_name(evidence.release, ROLLOUT_EXECUTOR_POOL_ID)
    uids_of_workload = _compute_pod_uids_of_workload(evidence.snapshots)

    for workload, carried in (
        (
            orchestrator,
            [
                ArgvManipulator.get_effective(command, flag)
                for command in _commands_of(evidence, workload=orchestrator, uids_of_workload=uids_of_workload)
            ],
        ),
        (
            rollout_executor,
            [
                _read_flag_of_serve_config(command, flag=flag)
                for command in _commands_of(evidence, workload=rollout_executor, uids_of_workload=uids_of_workload)
            ],
        ),
    ):
        assert len(carried) == len(values), (
            f"{workload} ran as {len(carried)} pod(s) while {len(values)} generation(s) of arguments were installed, "
            f"so the pods and the arguments cannot be paired up"
        )
        assert carried == list(values), (
            f"the successive pods of {workload} carried {flag} as {carried}, and the launches installed "
            f"{list(values)}: a take-over ran the component with arguments other than the ones it was relaunched with"
        )

    leaked = {
        uid: workload
        for workload, uids in uids_of_workload.items()
        if workload not in (orchestrator, rollout_executor)
        for uid in uids
        if any(value in part for part in evidence.commands_of_pod_uid[uid] for value in values)
    }
    assert not leaked, (
        f"{flag} is read by the rollout executor alone, and the pods {leaked} of other workloads carry one of its "
        f"values: the argument leaked into a payload a hot restart must leave untouched"
    )

    print(f"every take-over's orchestrator and rollout executor carried {flag} as relaunched: {list(values)}")


def _compute_pod_uids_of_workload(snapshots: Sequence[ClusterSnapshot]) -> dict[str, list[str]]:
    uids_of_workload: dict[str, list[str]] = {}
    for snapshot in snapshots:
        for pod in snapshot.pods:
            if (workload := _compute_workload_of_pod(pod.name, workloads=snapshot.workload_names)) is None:
                continue
            uids = uids_of_workload.setdefault(workload, [])
            if pod.uid not in uids:
                uids.append(pod.uid)
    return uids_of_workload


def _commands_of(
    evidence: HotRestartEvidence, *, workload: str, uids_of_workload: dict[str, list[str]]
) -> list[list[str]]:
    commands = []
    for uid in uids_of_workload.get(workload, []):
        assert (command := evidence.commands_of_pod_uid.get(uid)) is not None, (
            f"pod {uid} of {workload} was observed, but no read of the release recorded its command, so what it "
            f"ran is unknown"
        )
        commands.append(list(command))
    return commands


def _read_flag_of_serve_config(command: list[str], *, flag: str) -> str | None:
    assert (
        rendered := ArgvManipulator.get_effective(command, SERVE_CONFIG_FLAG)
    ) is not None, f"a served worker is started with {SERVE_CONFIG_FLAG}, and this command carries none: {command}"
    return parse_serve_worker_config(rendered).args.get(flag.removeprefix("--").replace("-", "_"))


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
