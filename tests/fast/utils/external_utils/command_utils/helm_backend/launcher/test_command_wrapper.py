import json
import subprocess

import pytest

from miles.utils.external_utils.command_utils.common import chart_dir
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH, ReleaseName
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL


class TestLogCommands:
    def test_follows_a_container_by_name(self):
        """A pod may gain a sidecar, and kubectl then refuses to guess which container to read."""
        command = Kubectl.logs_command(
            namespace="rl", target="statefulset/r-miles-run-orchestrator", container="orchestrator", follow=True
        )

        assert command[command.index("-c") + 1] == "orchestrator"
        assert "statefulset/r-miles-run-orchestrator" in command
        assert "--follow" in command

    def test_reads_every_container_of_a_pod_when_none_is_named(self):
        """A worker that crashed is the reason the run stopped, and its log may be in any of its containers."""
        command = Kubectl.logs_command(namespace="rl", target="pod/wb-0")

        assert "--all-containers" in command
        assert "-c" not in command
        assert "--follow" not in command

    def test_resumes_a_container_that_was_replaced(self):
        """The lines a crashed container wrote are the diagnosis, and only --previous still reaches them."""
        command = Kubectl.logs_command(
            namespace="rl", target="pod/wb-0", container="app", previous=True, since_time="2026-01-01T00:00:00Z"
        )

        assert "--previous" in command
        assert command[command.index("--since-time") + 1] == "2026-01-01T00:00:00Z"

    def test_selects_a_release_by_the_label_helm_stamps_on_every_pod_of_it(self):
        """An engine and a trainer share only this label, so anything narrower misses half the run."""
        assert Kubectl.release_selector("miles-run-x") == f"{INSTANCE_LABEL}=miles-run-x"

    def test_selects_a_job_by_the_label_the_job_controller_stamps_on_its_pods(self):
        """A command job's pods are named after their job, and only this label survives a pod restart."""
        assert Kubectl.job_selector("miles-run-command-gpu") == "batch.kubernetes.io/job-name=miles-run-command-gpu"


class TestUpgradeCommand:
    def test_installs_a_missing_release_and_updates_an_existing_one(self):
        """Relaunching a run id must update it in place, which plain upgrade would refuse to do."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=False)

        assert command[:4] == ["helm", "upgrade", "--install", "r"]

    def test_keeps_the_user_values_ahead_of_the_computed_ones(self):
        """A run value must win over a cluster default, and helm lets the later file win."""
        command = Helm.upgrade_command("r", "rl", "/c", ["/infra.yaml", "/run.yaml"], ci_run=False)

        assert command[command.index("/infra.yaml") - 1] == "--values"
        assert command.index("/infra.yaml") < command.index("/run.yaml")

    def test_labels_a_ci_release_so_the_next_run_can_clean_it_up(self):
        """The cleanup selects on this label, and an unlabelled CI release is one nothing will ever remove."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=True)

        assert command[command.index("--labels") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_leaves_a_human_release_unlabelled(self):
        """A developer's run carrying the CI label would be uninstalled by the next CI job in that namespace."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=False)

        assert "--labels" not in command

    def test_labels_the_release_rather_than_its_objects(self):
        """helm --labels records release metadata; a values-level label would not be selectable by helm list."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=True)

        assert command.index("--labels") > command.index("--namespace")


def _kubectl_answering(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = "") -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert argv[0] == "kubectl", f"only kubectl is expected to reach the process layer, got {argv[0]}"
        commands.append(argv[1:])
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, argv, stderr=stderr)
        return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    return commands


class TestCreateIfAbsent:
    def test_creates_the_objects_of_a_rendered_manifest(self, monkeypatch):
        """kubectl apply would adopt an object helm owns; create is what refuses to touch one."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stderr="")

        assert Kubectl.create_if_absent("/etc/miles/job.yaml")
        assert commands == [["create", "-f", "/etc/miles/job.yaml"]]

    def test_reports_an_object_that_was_already_there_without_failing(self, monkeypatch):
        """Its callers retry after a restart, and the whole point is that the second attempt is harmless."""
        _kubectl_answering(
            monkeypatch, returncode=1, stderr='Error from server (AlreadyExists): jobs "u" already exists'
        )

        assert not Kubectl.create_if_absent("/etc/miles/job.yaml")

    def test_refuses_to_read_any_other_failure_as_idempotence(self, monkeypatch):
        """A forbidden create means the object is missing, and pretending otherwise loses it silently."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="Error from server (Forbidden): jobs is forbidden")

        with pytest.raises(RuntimeError, match="Could not create"):
            Kubectl.create_if_absent("/etc/miles/job.yaml")


class TestDeleteJob:
    def test_treats_a_job_that_is_not_there_as_deleted(self, monkeypatch):
        """The launcher deletes a job that usually does not exist, which is the outcome it wants anyway."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stderr="")
        Kubectl.delete_job("miles-run-x-uninstall", namespace="rl", check=True)

        assert "--ignore-not-found" in commands[0]

    def test_lets_a_caller_that_cannot_go_on_without_the_deletion_fail(self, monkeypatch):
        """Installing over a job that is still armed hands the new release to the old run's uninstall."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="the api server refused")

        with pytest.raises(subprocess.CalledProcessError):
            Kubectl.delete_job("miles-run-x-uninstall", namespace="rl", check=True)

    def test_stays_tolerant_for_the_cleanup_of_a_command_job(self, monkeypatch):
        """That caller deletes the same job twice around a run, and neither call is worth failing over."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="the api server refused")

        Kubectl.delete_job("miles-run-command-convert", namespace="rl")


LAUNCHING_RUN_ID = "260101-000000-000"


def _recorded_ci_cleanup(
    monkeypatch: pytest.MonkeyPatch, namespace: str, *, listed: list[dict] | None = None
) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(listed or []), stderr="")

    monkeypatch.setattr(command_wrapper, "_run", fake_run)
    entrypoint._uninstall_leftover_ci_releases(namespace, keep_run_id=LAUNCHING_RUN_ID)
    return commands


class TestCiCleanup:
    def test_narrows_the_search_by_both_namespace_and_label(self, monkeypatch):
        """Deleting another user's run would kill a live experiment, so neither filter may be dropped."""
        command = _recorded_ci_cleanup(monkeypatch, "ci-runner-3")[0]

        assert command[command.index("--namespace") + 1] == "ci-runner-3"
        assert command[command.index("--selector") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_reads_the_release_names_helm_reports(self):
        """The names drive uninstall, so a parse that silently returns nothing would leave releases behind."""
        output = json.dumps([{"name": "miles-run-a", "namespace": "ci"}, {"name": "miles-run-b"}])

        assert [release["name"] for release in json.loads(output or "[]")] == ["miles-run-a", "miles-run-b"]

    def test_treats_no_output_as_nothing_to_clean(self):
        """helm prints nothing when no release matches, and that is not an error."""
        assert json.loads("" or "[]") == []

    def test_uninstalls_inside_the_namespace_it_was_told(self, monkeypatch):
        """A release name exists per namespace, so a missing namespace could hit a different one."""
        commands = _recorded_ci_cleanup(monkeypatch, "ci", listed=[{"name": "miles-run-a"}])

        assert commands[1] == ["helm", "uninstall", "miles-run-a", "--namespace", "ci"]


class TestChartDir:
    def test_finds_the_chart_inside_the_checkout(self):
        """The launcher installs the chart of the code it runs, not one from a registry."""
        assert chart_dir(repo_base_dir="/repo").as_posix() == "/repo/charts/miles-run"


LONGEST_RUN_ID = "a" * RUN_ID_MAX_LENGTH


def _unsplit(run_id: str) -> str:
    return ReleaseName(run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()


class TestReleaseName:
    def test_a_release_is_the_chart_name_the_run_id_and_the_component(self):
        """The launcher finds a run's release again from the run id alone, so the rule is fixed."""
        assert _unsplit("260101-000000-000") == "miles-run-260101-000000-000-all"

    def test_the_same_run_id_always_names_the_same_release(self):
        """Relaunching a run upgrades its release; a fresh name would deploy a second copy instead."""
        assert _unsplit(LONGEST_RUN_ID) == _unsplit(LONGEST_RUN_ID)


class TestComponentName:
    def test_an_object_is_the_release_the_chart_name_and_the_component(self):
        """Every object of a run is traceable to the release that made it."""
        assert naming.component_name("myrun", "orchestrator") == "myrun-miles-run-orchestrator"

    def test_a_release_that_already_carries_the_chart_name_is_not_told_it_twice(self):
        """The launcher's own releases start with the chart name, and doubling it wastes the budget."""
        assert naming.component_name("miles-run-260101", "orchestrator") == "miles-run-260101-orchestrator"

    def test_a_name_leaves_room_for_every_suffix_kubernetes_appends_below_it(self):
        """A pool name grows a cell index and then a revision hash, and a label value stops at 63."""
        name = naming.component_name("a" * 200, "orchestrator")
        appended = len(naming.LONGEST_CELL_INDEX_SUFFIX) + len(naming.LONGEST_REVISION_HASH_SUFFIX)

        assert len(name) + appended <= naming.MAX_OBJECT_NAME_LENGTH

    def test_the_component_survives_a_release_long_enough_to_fill_the_budget(self):
        """Truncating the component instead of the release would render two workloads under one name."""
        assert naming.component_name("a" * 200, "orchestrator").endswith("-orchestrator")

    def test_two_components_of_one_run_never_collapse_onto_the_same_name(self):
        """Truncating a name already at the limit silently merges two workloads into one object."""
        release = "a" * 200

        assert naming.component_name(release, "leader") != naming.component_name(release, "logger")

    def test_a_truncated_prefix_never_ends_on_the_separator(self):
        """A doubled dash is legal but reads as an empty segment, and drifts from the recorded names."""
        for length in range(1, 60):
            assert "--" not in naming.component_name("b" * length, "orchestrator")

    def test_a_component_longer_than_its_own_budget_is_hashed(self):
        """Letting it eat the whole budget leaves no room for the release digest, so two runs collide."""
        name = naming.component_name("a" * 200, "trainer-controller-" + "m" * 60)
        appended = len(naming.LONGEST_CELL_INDEX_SUFFIX) + len(naming.LONGEST_REVISION_HASH_SUFFIX)

        assert len(name) + appended <= naming.MAX_OBJECT_NAME_LENGTH

    def test_two_long_components_of_one_run_never_collapse_onto_the_same_name(self):
        """Truncation alone maps every long component onto one name; the digest is what keeps them apart."""
        prefix = "trainer-controller-" + "m" * 60

        assert naming.component_name("myrun", f"{prefix}-a") != naming.component_name("myrun", f"{prefix}-b")

    def test_a_long_component_leaves_the_release_its_digest(self):
        """Two releases whose names are truncated to the same prefix are only told apart by that digest."""
        long_component = "trainer-controller-" + "m" * 60

        assert naming.component_name("a" * 200, long_component) != naming.component_name("b" * 200, long_component)

    def test_the_same_release_and_component_always_name_the_same_object(self):
        """helm upgrade replaces an object in place only while its name is unchanged."""
        assert naming.component_name("miles-run-x", "trainer-engine-actor") == naming.component_name(
            "miles-run-x", "trainer-engine-actor"
        )


class TestStaticWorkerHost:
    def test_a_static_cell_is_reached_through_its_own_pod_of_the_headless_service(self):
        """A pool of session servers is several addresses, and pod zero can answer only one of them."""
        assert naming.static_worker_host("myrun", "session-server", 1) == (
            "myrun-miles-run-session-server-1.myrun-miles-run-session-server"
        )
