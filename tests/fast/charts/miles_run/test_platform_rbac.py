import json
from typing import Any

from tests.fast.charts.utils import (
    RUN_ORCHESTRATOR_NAME,
    RUN_PLATFORM_READ_DELETE_NAME,
    RUN_PLATFORM_READ_NAME,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    requires_helm,
    with_object_names,
)

_READER = {
    "name": "inference-registration-reporter",
    "command": ["x"],
    "serviceAccountName": RUN_PLATFORM_READ_NAME,
}
_DELETER = {
    "name": "trainer-controller-actor",
    "command": ["x"],
    "serviceAccountName": RUN_PLATFORM_READ_DELETE_NAME,
}
_PLAIN = {"name": "inference-engine", "command": ["x"]}


def _render_without_orchestrator(*workers: dict[str, Any]) -> list[dict[str, Any]]:
    return render_run(
        "--set-json",
        "run.orchestrator.command=[]",
        "--set-json",
        f"run.staticWorkers={json.dumps(with_object_names(list(workers)))}",
    )


@requires_helm
class TestTheAccountsAReleaseGrantsItsPlatformClients:
    def test_grants_the_read_account_to_a_release_whose_client_only_reads_pods(self):
        """A split run's inference release carries the registration reporter, which lists pods to report them."""
        objects = _render_without_orchestrator(_READER)

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_NAME)
        assert named_object(objects, "RoleBinding", RUN_PLATFORM_READ_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}
        ]
        assert (
            pod_spec_of(objects, "StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-inference-registration-reporter")[
                "serviceAccountName"
            ]
            == RUN_PLATFORM_READ_NAME
        )

    def test_grants_the_read_delete_account_to_a_release_whose_client_also_deletes_pods(self):
        """A split run's trainer release suspends cells by deleting pods, which the read-only account refuses."""
        objects = _render_without_orchestrator(_DELETER)

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "RoleBinding", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_DELETE_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]

    def test_grants_a_reader_none_of_the_rights_the_read_delete_account_carries(self):
        """A reporter may read pod state without inheriting deletion or job-creation rights."""
        objects = _render_without_orchestrator(_READER)

        assert not any(
            obj["metadata"]["name"] == RUN_PLATFORM_READ_DELETE_NAME for obj in objects_of_kind(objects, "Role")
        )

    def test_grants_nothing_to_a_release_no_platform_client_lives_in(self):
        """A release of plain engines binds no account, so rights over pods would be handed to nobody."""
        objects = _render_without_orchestrator(_PLAIN)

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []

    def test_still_grants_the_release_that_runs_the_orchestration_script(self):
        """The orchestrator deletes pods to heal cells, and it is the reason this rbac existed at all."""
        objects = render_run()

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_DELETE_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
        assert (
            pod_spec_of(objects, "StatefulSet", RUN_ORCHESTRATOR_NAME)["serviceAccountName"]
            == RUN_PLATFORM_READ_DELETE_NAME
        )
