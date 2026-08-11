import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.common import chart_dir
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.worker_provider.kubernetes.helm import naming


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


class TestUpgradeCommand:
    def test_installs_a_missing_release_and_updates_an_existing_one_without_ci_run_argument(self):
        """Relaunching a run id must update it in place, which plain upgrade would refuse to do."""
        command = Helm.upgrade_command("r", "myns", "/c", [])

        assert command[:4] == ["helm", "upgrade", "--install", "r"]

    def test_keeps_the_user_values_ahead_of_the_computed_ones_without_ci_run_argument(self):
        """A run value must win over a cluster default, and helm lets the later file win."""
        command = Helm.upgrade_command("r", "myns", "/c", ["/infra.yaml", "/run.yaml"])

        assert command[command.index("/infra.yaml") - 1] == "--values"
        assert command.index("/infra.yaml") < command.index("/run.yaml")


class TestBuildDependencies:
    def test_chart_dependencies_are_rebuilt_only_when_a_locked_dependency_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Helm rebuilds a locked chart only after one of its cached dependencies disappears."""
        chart = tmp_path / "chart"
        charts = chart / "charts"
        charts.mkdir(parents=True)
        (chart / "Chart.lock").write_text("dependencies:\n  - name: worker\n  - name: runtime\n")
        (charts / "worker").mkdir()
        runtime = charts / "runtime"
        runtime.mkdir()
        commands: list[list[str]] = []

        def fake_run_process(
            argv: list[str],
            *,
            capture_output: bool,
            check: bool,
            input: str | None = None,
            timeout: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            commands.append(argv)
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run_process)

        Helm.build_dependencies(chart)
        runtime.rmdir()
        Helm.build_dependencies(chart)

        assert commands == [["helm", "dependency", "build", str(chart)]]


class TestRawCommands:
    def test_a_helm_call_reports_its_failure_instead_of_raising(self, monkeypatch):
        """The callers of these wrappers all want to read a failure, not to be unwound by it."""
        recorded = {}

        def fake_run(argv, *, capture_output, check, input=None):
            recorded.update(argv=argv, check=check)
            return None

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper.run_process", fake_run
        )
        Helm.run_raw("template", "r", "/c")

        assert recorded["argv"] == ["helm", "template", "r", "/c"]
        assert recorded["check"] is False

    def test_a_kubectl_call_is_spelled_out_the_same_way(self, monkeypatch):
        """One wrapper for both binaries is what keeps command strings out of the callers."""
        recorded = {}

        def fake_run(argv, *, capture_output, check, input=None):
            recorded.update(argv=argv)
            return None

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper.run_process", fake_run
        )
        Kubectl.run_raw("get", "namespace", "--", "myns")

        assert recorded["argv"] == ["kubectl", "get", "namespace", "--", "myns"]


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


class TestGetJson:
    def test_a_failed_get_is_not_reported_as_an_absent_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed lookup must expose its exit code and stderr instead of looking like an absent object."""
        _kubectl_answering(monkeypatch, returncode=23, stderr="the api server refused the request")

        with pytest.raises(RuntimeError, match="code 23: the api server refused the request"):
            Kubectl.get_json("pod", return_type=Pod, name="trainer-0", namespace="rl")


class TestChartDir:
    def test_finds_the_chart_inside_the_checkout(self):
        """The launcher installs the chart of the code it runs, not one from a registry."""
        assert chart_dir(repo_base_dir="/repo").as_posix() == "/repo/charts/miles-run"


class TestReleaseName:
    def test_a_release_is_the_chart_name_and_the_run_id(self):
        """The launcher finds a run's release again from the run id alone, so the rule is fixed."""
        assert RunNames.release(run_id="260101-000000-000") == "miles-run-260101-000000-000"

    def test_the_same_run_id_always_names_the_same_release(self):
        """Relaunching a run upgrades its release; a fresh name would deploy a second copy instead."""
        run_id = "a" * 32

        assert RunNames.release(run_id=run_id) == RunNames.release(run_id=run_id)


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
