import json
import subprocess

import pytest
import yaml
from tests.e2e.deploy.conftest_deploy.hot_restart import guarded_launcher
from tests.e2e.deploy.conftest_deploy.hot_restart.guard_manifest import guard_manifest
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_diff import diff_manifests
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


@pytest.mark.parametrize("change_template", [False, True])
def test_a_second_guarded_upgrade_ignores_only_guard_metadata(change_template: bool) -> None:
    """Stored identity preconditions must permit another takeover without hiding trainer changes."""
    target = SoakDeploymentTarget(
        namespace="ns",
        release="release",
        workload_uids={"trainer": "uid"},
        workload_stamps={"trainer": None},
        saved_iteration=1,
        finished_rollout_id=2,
    )
    rendered = yaml.safe_dump(
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": "trainer", "namespace": "ns"},
            "spec": {"template": {"spec": {"containers": [{"name": "trainer", "image": "original"}]}}},
        }
    )
    stored = Manifest.parse(
        guard_manifest(
            rendered=rendered,
            target=target,
            payloads={
                "statefulsets": {
                    "items": [
                        {"kind": "StatefulSet", "metadata": {"name": "trainer", "uid": "uid", "resourceVersion": "42"}}
                    ]
                }
            },
        ),
        namespace="ns",
    )
    after = Manifest.parse(rendered.replace("original", "changed") if change_template else rendered, namespace="ns")

    assert not diff_manifests(before=stored, after=after).is_allowed
    normalized = guarded_launcher._without_guard_preconditions(stored)
    assert diff_manifests(before=normalized, after=after).is_allowed is (not change_template)
    assert stored.objects[0].body["metadata"]["uid"] == "uid"
    assert stored.objects[0].body["metadata"]["resourceVersion"] == "42"


@pytest.mark.parametrize("current_uid", [None, "observed", "replacement"])
def test_cleanup_only_deletes_the_observed_job(monkeypatch: pytest.MonkeyPatch, current_uid: str | None) -> None:
    """Cleanup must carry the observed UID to the delete endpoint and reject replacement jobs."""
    deletes: list[dict] = []

    def command(
        argv: list[str], *, capture_output: bool, check: bool, input: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if argv[1] == "get":
            payload = json.dumps({"metadata": {"uid": current_uid}}) if current_uid else ""
        else:
            payload = ""
            if argv[1] == "delete":
                assert "--raw" in argv and input is not None
                deletes.append(json.loads(input))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(guarded_launcher, "run_process", command)
    if current_uid == "observed":
        guarded_launcher._delete_observed_job(name="uninstall", namespace="ns", expected_uid="observed")
        assert len(deletes) == 1
        assert deletes[0]["preconditions"] == {"uid": "observed"}
        assert deletes[0]["propagationPolicy"] == "Foreground"
    else:
        with pytest.raises(StaleFaultTargetError):
            guarded_launcher._delete_observed_job(name="uninstall", namespace="ns", expected_uid="observed")
        assert deletes == []
