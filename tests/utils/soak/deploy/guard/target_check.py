import json

from tests.utils.deploy.hot_restart.cluster_observer import WORKLOAD_KINDS, parse_workload_facts
from tests.utils.soak.deploy.types import DeploymentTarget

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


def assert_workloads_unchanged(target: DeploymentTarget) -> None:
    uids: dict[str, str] = {}
    stamps: dict[str, str | None] = {}
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
            if metadata.get("deletionTimestamp") is not None or name in uids:
                raise StaleFaultTargetError(f"Deployment workload {name} is deleting or ambiguous")
            if not (uid := metadata.get("uid")):
                raise StaleFaultTargetError(f"Deployment workload {name} has no uid")
            uids[name] = uid
            stamps[name] = restart_at_of_name.get(name)

    if not uids or uids != target.workload_uids or stamps != target.workload_stamps:
        raise StaleFaultTargetError(f"Deployment {target.namespace}/{target.release} changed since observation")
