import subprocess

from tests.fast.charts.utils import (
    CHART_DIR,
    NAMESPACE,
    RELEASE_NAME,
    UNINSTALLER_SERVICE_ACCOUNT,
    named_object,
    render,
    requires_helm,
)

_OBJECT_NAME = "rendered-workbench"
_WORKBENCH_SERVICE_ACCOUNT = "workbench-operator"
_COMMON_OVERRIDES = (
    "--set",
    f"objectName={_OBJECT_NAME}",
    "--set",
    f"serviceAccount.name={_WORKBENCH_SERVICE_ACCOUNT}",
)


def _render_notes(*args: str) -> str:
    result = subprocess.run(
        [
            "helm",
            "install",
            RELEASE_NAME,
            str(CHART_DIR),
            "--namespace",
            NAMESPACE,
            "--dry-run=client",
            *_COMMON_OVERRIDES,
            *args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.rpartition("NOTES:\n")[2]


@requires_helm
class TestNotes:
    def test_notes_render_actionable_commands_and_only_the_disabled_rbac_warnings(self):
        """Actionable commands and account names stay exact while RBAC warnings follow disabled capabilities."""
        objects = render(*_COMMON_OVERRIDES)
        default_notes = _render_notes()
        no_rbac_notes = _render_notes("--set", "rbac.create=false")
        no_lws_notes = _render_notes("--set", "rbac.leaderWorkerSets=false")

        statefulset_name = named_object(objects, "StatefulSet", _OBJECT_NAME)["metadata"]["name"]
        workbench_account = named_object(objects, "ServiceAccount", _WORKBENCH_SERVICE_ACCOUNT)["metadata"]["name"]
        uninstaller_account = named_object(objects, "ServiceAccount", UNINSTALLER_SERVICE_ACCOUNT)["metadata"]["name"]

        for notes in (default_notes, no_rbac_notes, no_lws_notes):
            assert f"kubectl exec -it statefulset/{statefulset_name} -n {NAMESPACE} -- bash" in notes
            assert f'ServiceAccount "{workbench_account}"' in notes
            assert f'ServiceAccount "{uninstaller_account}"' in notes
            assert f"helm uninstall {RELEASE_NAME} -n {NAMESPACE}" in notes

        assert default_notes.count("rbac.create=false:") == 0
        assert default_notes.count("rbac.leaderWorkerSets=false:") == 0
        assert no_rbac_notes.count("rbac.create=false:") == 1
        assert no_rbac_notes.count("rbac.leaderWorkerSets=false:") == 0
        assert no_lws_notes.count("rbac.create=false:") == 0
        assert no_lws_notes.count("rbac.leaderWorkerSets=false:") == 1
