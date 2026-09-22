import json
from typing import NamedTuple

from tests.utils.deploy.hot_restart.cluster_observer import WORKLOAD_KINDS, parse_workload_facts
from tests.utils.soak.deploy.types import DeploymentTarget

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


class ObservedWorkloadKey(NamedTuple):
    kind: str
    name: str


class ObservedWorkload(FrozenStrictBaseModel):
    kind: str
    name: str
    uid: str
    resource_version: str
    restart_at: str | None


def read_validated_workloads(target: DeploymentTarget) -> dict[ObservedWorkloadKey, ObservedWorkload]:
    observed: dict[ObservedWorkloadKey, ObservedWorkload] = {}
    for kind in WORKLOAD_KINDS:
        payload = json.loads(
            run_process(
                [
                    "kubectl",
                    "get",
                    kind,
                    "--namespace",
                    target.namespace,
                    "--selector",
                    compute_release_selector(release=target.release),
                    "--output",
                    "json",
                ],
                capture_output=True,
                check=True,
                timeout=KUBECTL_TIMEOUT_SECONDS,
            ).stdout
        )
        restart_at_of_name = {fact.name: fact.restart_at for fact in parse_workload_facts(payload, kind=kind)}
        for item in payload["items"]:
            metadata = item["metadata"]
            name = metadata["name"]
            if metadata.get("deletionTimestamp") is not None or any(key.name == name for key in observed):
                raise StaleFaultTargetError(f"Deployment workload {name} is deleting or ambiguous")
            if not (uid := metadata.get("uid")):
                raise StaleFaultTargetError(f"Deployment workload {name} has no uid")
            observed[ObservedWorkloadKey(kind=item["kind"], name=name)] = ObservedWorkload(
                kind=item["kind"],
                name=name,
                uid=uid,
                resource_version=metadata.get("resourceVersion") or "",
                restart_at=restart_at_of_name.get(name),
            )

    _validate_workloads(target=target, observed=observed)
    return observed


def _validate_workloads(*, target: DeploymentTarget, observed: dict[ObservedWorkloadKey, ObservedWorkload]) -> None:
    uids = {workload.name: workload.uid for workload in observed.values()}
    stamps = {workload.name: workload.restart_at for workload in observed.values()}
    if not uids or uids != target.workload_uids or stamps != target.workload_stamps:
        raise StaleFaultTargetError(f"Deployment {target.namespace}/{target.release} changed since observation")
