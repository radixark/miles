import asyncio
import json

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import WORKLOAD_KINDS, parse_workload_facts
from tests.utils.soak.action import run_command
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, compute_release_selector
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


async def validate_deployment_target(target: SoakDeploymentTarget) -> None:
    await read_validated_workloads(target)


async def read_validated_workloads(target: SoakDeploymentTarget) -> dict[str, dict]:
    results = await asyncio.gather(
        *(
            run_command(
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
                timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
            )
            for kind in WORKLOAD_KINDS
        )
    )
    payloads = {kind: json.loads(result.stdout) for kind, result in zip(WORKLOAD_KINDS, results, strict=True)}
    _validate_workloads(target=target, payloads=payloads)
    return payloads


def _validate_workloads(*, target: SoakDeploymentTarget, payloads: dict[str, dict]) -> None:
    uids = {}
    stamps = {}
    for kind in WORKLOAD_KINDS:
        payload = payloads[kind]
        for item in payload["items"]:
            metadata = item["metadata"]
            name = metadata["name"]
            if metadata.get("deletionTimestamp") is not None or name in uids:
                raise StaleFaultTargetError(f"Deployment workload {name} is deleting or ambiguous")
            uids[name] = metadata["uid"]
        stamps.update({fact.name: fact.restart_at for fact in parse_workload_facts(payload, kind=kind)})
    if not uids or uids != target.workload_uids or stamps != target.workload_stamps:
        raise StaleFaultTargetError(f"Deployment {target.namespace}/{target.release} changed since observation")
