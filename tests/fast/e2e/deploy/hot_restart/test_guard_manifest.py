import pytest
import yaml
from tests.e2e.deploy.conftest_deploy.hot_restart.guard_manifest import guard_manifest
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.workers.cell_operations.base import StaleFaultTargetError


@pytest.mark.parametrize("changed", [None, "uid", "missing", "extra", "namespace"])
def test_rendered_upgrade_keeps_preconditions_and_rejects_unobserved_workloads(changed: str | None) -> None:
    """Every submitted workload carries the observed UID and the validated resource version."""
    target = SoakDeploymentTarget(
        namespace="ns",
        release="release",
        workload_uids={"worker": "uid"},
        workload_stamps={"worker": "stamp"},
        saved_iteration=1,
        finished_rollout_id=2,
    )
    payloads = {
        "statefulsets": {
            "items": [
                {
                    "kind": "StatefulSet",
                    "metadata": {
                        "name": "worker",
                        "uid": "replacement" if changed == "uid" else "uid",
                        "resourceVersion": "42",
                    },
                }
            ]
        }
    }
    documents = [
        {
            "kind": "StatefulSet",
            "metadata": {"name": "worker", "namespace": "other" if changed == "namespace" else "ns"},
            "spec": {"replicas": 2},
        }
    ]
    if changed == "missing":
        documents = []
    if changed == "extra":
        documents.append({"kind": "StatefulSet", "metadata": {"name": "extra"}})
    if changed is None:
        output = list(
            yaml.safe_load_all(
                guard_manifest(rendered=yaml.safe_dump_all(documents), target=target, payloads=payloads)
            )
        )
        assert output[0]["metadata"] == {"name": "worker", "namespace": "ns", "uid": "uid", "resourceVersion": "42"}
        assert output[0]["spec"] == {"replicas": 2}
    else:
        with pytest.raises(StaleFaultTargetError):
            guard_manifest(rendered=yaml.safe_dump_all(documents), target=target, payloads=payloads)
