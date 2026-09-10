import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.deployment_target import _validate_workloads
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


@pytest.mark.parametrize("changed", [None, "uid", "stamp", "gone", "deleting"])
def test_a_replaced_or_disappeared_deployment_is_rejected(changed: str | None) -> None:
    """A queued hot restart may only enter against its observed workload identities."""
    target = SoakDeploymentTarget(
        namespace="ns",
        release="release",
        workload_uids={"worker": "uid"},
        workload_stamps={"worker": "stamp"},
        saved_iteration=1,
        finished_rollout_id=2,
    )
    metadata = {"name": "worker", "uid": "new" if changed == "uid" else "uid", "generation": 1}
    if changed == "deleting":
        metadata["deletionTimestamp"] = "2026-09-11T00:00:00Z"
    item = {
        "metadata": metadata,
        "spec": {
            "template": {
                "metadata": {"annotations": {RESTART_AT_ANNOTATION: "new" if changed == "stamp" else "stamp"}}
            }
        },
    }
    payloads = {
        "statefulsets": {"items": [] if changed == "gone" else [item]},
        "leaderworkersets.leaderworkerset.x-k8s.io": {"items": []},
    }

    if changed is None:
        _validate_workloads(target=target, payloads=payloads)
    else:
        with pytest.raises(StaleFaultTargetError):
            _validate_workloads(target=target, payloads=payloads)
