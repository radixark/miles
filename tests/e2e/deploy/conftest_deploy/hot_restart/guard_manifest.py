import asyncio
import sys
from pathlib import Path

import typer
import yaml
from tests.e2e.deploy.conftest_deploy.hot_restart.deployment_target import read_validated_workloads
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    LEADER_WORKER_SET_KIND,
    STATEFUL_SET_KIND,
)
from miles.utils.workers.cell_operations.base import StaleFaultTargetError

GUARDED_WORKLOAD_KINDS = frozenset({STATEFUL_SET_KIND, LEADER_WORKER_SET_KIND})


def guard_manifest(*, rendered: str, target: SoakDeploymentTarget, payloads: dict[str, dict]) -> str:
    observed = {
        (item["kind"], item["metadata"]["name"]): item for payload in payloads.values() for item in payload["items"]
    }
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    guarded = set()
    for document in documents:
        metadata = document["metadata"]
        key = (document["kind"], metadata["name"])
        if key not in observed:
            if document["kind"] in GUARDED_WORKLOAD_KINDS:
                raise StaleFaultTargetError(f"Unobserved rendered workload {key}")
            continue
        if key in guarded or metadata.get("namespace", target.namespace) != target.namespace:
            raise StaleFaultTargetError(f"Ambiguous rendered workload {key}")
        current = observed[key]["metadata"]
        if current["uid"] != target.workload_uids[key[1]] or not current["resourceVersion"]:
            raise StaleFaultTargetError(f"Changed rendered workload {key}")
        metadata["uid"] = current["uid"]
        metadata["resourceVersion"] = current["resourceVersion"]
        guarded.add(key)
    if guarded != set(observed):
        raise StaleFaultTargetError("The upgrade drops observed workloads")
    return yaml.safe_dump_all(documents, sort_keys=False)


app = typer.Typer()


@app.command()
def main(target_path: Path) -> None:
    target = SoakDeploymentTarget.model_validate_json(target_path.read_text())
    payloads = asyncio.run(read_validated_workloads(target))
    sys.stdout.write(guard_manifest(rendered=sys.stdin.read(), target=target, payloads=payloads))


if __name__ == "__main__":
    app()
