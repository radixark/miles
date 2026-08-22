from functools import cache
from typing import Any

import yaml
from tests.fast.charts.utils import (
    CHART_DIR,
    NAMESPACE,
    RUN_CHART_DIR,
    RUN_ORCHESTRATOR_NAME,
    RUN_RELEASE_NAME,
    RUN_UNINSTALL_JOB_NAME,
    RUN_UNINSTALL_MANIFEST_NAME,
    UNINSTALLER_SERVICE_ACCOUNT,
    named_object,
    objects_of_kind,
    only_container_of,
    render_run,
    requires_helm,
)

from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

MANIFEST_PATH = "/etc/miles-uninstall/uninstall-job.yaml"

ENABLE_AUTO_UNINSTALL = ("--set", "run.autoUninstall.enabled=true")
DISABLE_AUTO_UNINSTALL = ("--set", "run.autoUninstall.enabled=false")


def _enabled_objects(*args: str) -> list[dict[str, Any]]:
    if args:
        return render_run(*ENABLE_AUTO_UNINSTALL, *args)
    return list(_default_enabled_objects())


@cache
def _default_enabled_objects() -> tuple[dict[str, Any], ...]:
    return tuple(render_run(*ENABLE_AUTO_UNINSTALL))


def _rendered_job(*args: str) -> dict[str, Any]:
    config_map = named_object(_enabled_objects(*args), "ConfigMap", RUN_UNINSTALL_MANIFEST_NAME)
    return yaml.safe_load(config_map["data"]["uninstall-job.yaml"])


@requires_helm
class TestUninstallManifest:
    def test_leaves_the_logs_readable_for_two_minutes_before_uninstalling_the_release(self):
        """The wrapper creates the job the moment the verdict is in, so the delay has to live in the job."""
        container = _rendered_job()["spec"]["template"]["spec"]["containers"][0]

        assert container["command"] == [
            "sh",
            "-c",
            f"sleep 120 && helm uninstall {RUN_RELEASE_NAME} --namespace myns --ignore-not-found",
        ]

    def test_treats_an_already_gone_release_as_a_successful_uninstall(self):
        """Split runs uninstall their own release, so this job routinely finds nothing and must not fail."""
        command = _rendered_job()["spec"]["template"]["spec"]["containers"][0]["command"][-1]

        assert "--ignore-not-found" in command

    def test_runs_as_the_account_that_outlives_the_release(self):
        """helm deletes the release's own rolebinding halfway through, which would 403 the rest of the deletions."""
        assert _rendered_job()["spec"]["template"]["spec"]["serviceAccountName"] == UNINSTALLER_SERVICE_ACCOUNT

    def test_carries_no_label_the_release_selector_matches(self):
        """The launcher follows and deletes objects by that selector, and would take the escape job with them."""
        job = _rendered_job()

        assert INSTANCE_LABEL not in job["metadata"]["labels"]
        assert INSTANCE_LABEL not in job["spec"]["template"]["metadata"]["labels"]
        assert job["metadata"]["labels"] == {"miles.radixark.io/uninstall-of": RUN_RELEASE_NAME}

    def test_names_the_job_what_the_launcher_deletes_on_a_relaunch(self):
        """Defusing a pending uninstall needs the name python computed, not one the chart invented."""
        metadata = _rendered_job()["metadata"]

        assert (metadata["name"], metadata["namespace"]) == (RUN_UNINSTALL_JOB_NAME, NAMESPACE)

    def test_gives_up_and_collects_itself_instead_of_waiting_forever(self):
        """A job that never got its node must not hold a slot, and its corpse must not outlive the namespace."""
        spec = _rendered_job()["spec"]

        assert (spec["activeDeadlineSeconds"], spec["ttlSecondsAfterFinished"], spec["backoffLimit"]) == (
            1020,
            3600,
            2,
        )

    def test_schedules_where_every_other_pod_of_the_run_may_run(self):
        """A cluster that needs a toleration needs it here too, or the release is never uninstalled."""
        job = _rendered_job("--set", "infra.scheduling.tolerations[0].key=gpu")

        assert job["spec"]["template"]["spec"]["tolerations"] == [{"key": "gpu"}]

    def test_stays_out_of_the_release_it_uninstalls(self):
        """A job helm owns would be deleted by the very uninstall it is running."""
        assert objects_of_kind(_enabled_objects(), "Job") == []


@requires_helm
class TestOrchestratorTrigger:
    def test_hands_the_wrapper_the_manifest_it_creates_the_job_from(self):
        """The wrapper is the first authority on a finished run, and it can only apply a file it was told about."""
        container = only_container_of(_enabled_objects(), "StatefulSet", RUN_ORCHESTRATOR_NAME)
        separator = container["command"].index("--")

        assert container["command"][separator - 2 : separator] == ["--uninstall-manifest", MANIFEST_PATH]
        assert {"name": "uninstall-manifest", "mountPath": "/etc/miles-uninstall", "readOnly": True} in container[
            "volumeMounts"
        ]

    def test_lets_the_orchestrator_create_that_one_job(self):
        """The wrapper creates it as the run's own account, which is otherwise allowed pods and nothing else."""
        role = named_object(_enabled_objects(), "Role", RUN_ORCHESTRATOR_NAME)

        assert {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]} in role["rules"]

    def test_grants_no_job_rights_to_a_run_that_never_creates_one(self):
        """A run that cannot uninstall itself has no business creating workloads of any kind."""
        role = named_object(render_run(*DISABLE_AUTO_UNINSTALL), "Role", RUN_ORCHESTRATOR_NAME)

        assert [rule["apiGroups"] for rule in role["rules"]] == [[""]]

    def test_says_nothing_about_uninstalling_when_the_launcher_found_no_uninstaller(self):
        """A namespace without that account cannot half-uninstall a release, so nothing about it is rendered."""
        objects = render_run(*DISABLE_AUTO_UNINSTALL)

        assert (
            "--uninstall-manifest" not in only_container_of(objects, "StatefulSet", RUN_ORCHESTRATOR_NAME)["command"]
        )
        assert objects_of_kind(objects, "ConfigMap") == []


class TestUninstallerAccountName:
    def test_a_bare_helm_install_cleans_up_after_itself_too(self):
        """The launcher always writes the flag, so this default is what a hand-installed release gets."""
        run_default = yaml.safe_load((RUN_CHART_DIR / "values.yaml").read_text())["run"]["autoUninstall"]

        assert run_default["enabled"] is True

    def test_both_charts_default_to_the_one_name_python_knows(self):
        """The workbench creates that account and a run's job runs as it, so a drift would 403 every uninstall."""
        run_default = yaml.safe_load((RUN_CHART_DIR / "values.yaml").read_text())["run"]["autoUninstall"]
        workbench_default = yaml.safe_load((CHART_DIR / "values.yaml").read_text())["uninstaller"]

        assert run_default["serviceAccount"] == UNINSTALLER_SERVICE_ACCOUNT
        assert workbench_default["serviceAccount"] == UNINSTALLER_SERVICE_ACCOUNT

    def test_the_tests_and_the_code_agree_on_that_name(self):
        """Every assertion here is written against the constant the launcher and the charts actually use."""
        assert UNINSTALLER_SERVICE_ACCOUNT == naming.UNINSTALLER_SERVICE_ACCOUNT
