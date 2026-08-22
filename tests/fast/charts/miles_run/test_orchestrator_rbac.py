import json
from typing import Any

from tests.fast.charts.utils import (
    RUN_ORCHESTRATOR_NAME,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    requires_helm,
    with_object_names,
)

_PLATFORM_READER_NAME = f"{RUN_RELEASE_NAME}-miles-run-platform-reader"
_READER = {
    "name": "inference-registration-reporter",
    "command": ["x"],
    "serviceAccountName": _PLATFORM_READER_NAME,
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
class TestTheRbacThePodReadersNeed:
    def test_grants_a_release_that_runs_no_orchestration_script_but_reads_pods(self):
        """A split run's inference release carries the registration reporter, which lists pods to report them."""
        objects = _render_without_orchestrator(_READER)

        assert named_object(objects, "ServiceAccount", _PLATFORM_READER_NAME)
        assert named_object(objects, "Role", _PLATFORM_READER_NAME)
        assert named_object(objects, "RoleBinding", _PLATFORM_READER_NAME)
        assert (
            pod_spec_of(objects, "StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-inference-registration-reporter")[
                "serviceAccountName"
            ]
            == _PLATFORM_READER_NAME
        )

    def test_grants_only_the_pod_verbs_the_reporter_calls(self):
        """A reporter may read pod state without inheriting deletion or job-creation rights."""
        objects = _render_without_orchestrator(_READER)
        role = named_object(objects, "Role", _PLATFORM_READER_NAME)

        assert role["rules"] == [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}]
        assert not any(obj["metadata"]["name"] == RUN_ORCHESTRATOR_NAME for obj in objects_of_kind(objects, "Role"))

    def test_grants_nothing_to_a_release_no_pod_reader_lives_in(self):
        """A trainer release binds no service account, so rights to delete pods would be handed to nobody."""
        objects = _render_without_orchestrator(_PLAIN)

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []

    def test_still_grants_the_release_that_runs_the_orchestration_script(self):
        """The orchestrator deletes pods to heal cells, and it is the reason this rbac existed at all."""
        objects = render_run()

        assert named_object(objects, "ServiceAccount", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "Role", RUN_ORCHESTRATOR_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
