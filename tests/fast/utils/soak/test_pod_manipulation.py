import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.fault_forms import DeletePodFaultForm
from tests.utils.soak.state import SoakActionRequest, SoakPodTarget

from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import DeployComponent

_RELEASE = ReleaseName(run_id="abc123", deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()
_NAMESPACE = "miles-e2e"


class _ApiError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status


@pytest.mark.parametrize(
    "outcome",
    ["replaced", "missing", "already_deleting", "read_failed", "other_release", "other_namespace", "stale_uid"],
)
async def test_pod_deletion_requires_a_new_delete_and_confirmed_old_uid_absence(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """An existing deletion or failed observation cannot count as an applied fault."""
    monkeypatch.setitem(
        sys.modules,
        "kubernetes_asyncio",
        SimpleNamespace(
            client=SimpleNamespace(
                V1DeleteOptions=SimpleNamespace,
                V1Preconditions=SimpleNamespace,
                ApiException=_ApiError,
                ApiClient=lambda: AsyncMock(),
                CoreV1Api=lambda client: api,
            ),
            config=SimpleNamespace(load_incluster_config=lambda: None),
        ),
    )
    before = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="stale" if outcome == "stale_uid" else "uid",
            resource_version="rv",
            deletion_timestamp="deleting" if outcome == "already_deleting" else None,
        )
    )
    after = (
        SimpleNamespace(metadata=SimpleNamespace(uid="replacement"))
        if outcome == "replaced"
        else _ApiError(404 if outcome == "missing" else 500)
    )
    api = SimpleNamespace(
        read_namespaced_pod=AsyncMock(side_effect=[before, after]), delete_namespaced_pod=AsyncMock()
    )
    pod = SoakPodTarget(
        namespace="other" if outcome == "other_namespace" else _NAMESPACE,
        release="other" if outcome == "other_release" else _RELEASE,
        name="pod",
        uid="uid",
    )
    form = DeletePodFaultForm(namespace=_NAMESPACE, run_id="abc123")
    request = SoakActionRequest(target=typed_cell("actor-3", "actor"), form_name=form.name, harms_cell=True, pod=pod)

    if outcome in {"replaced", "missing"}:
        assert await form.execute(request) == {
            "kind": "pod_deleted",
            "namespace": _NAMESPACE,
            "pod_name": "pod",
            "pod_uid": "uid",
        }
    else:
        with pytest.raises(_ApiError if outcome == "read_failed" else AssertionError):
            await form.execute(request)
    if outcome in {"already_deleting", "other_release", "other_namespace", "stale_uid"}:
        api.delete_namespaced_pod.assert_not_called()
    else:
        assert api.delete_namespaced_pod.await_count == 1
        preconditions = api.delete_namespaced_pod.call_args.kwargs["body"].preconditions
        assert preconditions.uid == "uid" and preconditions.resource_version == "rv"
        assert api.delete_namespaced_pod.call_args.kwargs["namespace"] == _NAMESPACE
        assert api.delete_namespaced_pod.call_args.kwargs["name"] == "pod"
