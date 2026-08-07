import os
import subprocess
import sys

import pytest

from tests.fast.charts.utils import CHART_DIR, CLI_PATH


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "calls.log"
    failing_path = tmp_path / "failing"
    failing_path.write_text("")
    refused_path = tmp_path / "refused"
    refused_path.write_text("")

    for binary in ("helm", "kubectl"):
        (bin_dir / binary).write_text(
            "#!/usr/bin/env bash\n"
            f'command="{binary} $@"\n'
            f'echo "$command" >> {calls_path}\n'
            'while IFS= read -r line || [ -n "$line" ]; do\n'
            '  [ -z "$line" ] && continue\n'
            '  status="${line%%|*}"\n'
            '  pattern="${line#*|}"\n'
            '  case "$command" in\n'
            '    $pattern*) echo "Error from server ($status): refused" >&2; exit 1 ;;\n'
            "  esac\n"
            f"done < {refused_path}\n"
            'while IFS= read -r pattern || [ -n "$pattern" ]; do\n'
            '  [ -z "$pattern" ] && continue\n'
            '  case "$command" in\n'
            "    $pattern*) exit 1 ;;\n"
            "  esac\n"
            f"done < {failing_path}\n"
            "exit 0\n"
        )
        (bin_dir / binary).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    return dict(calls_path=calls_path, failing_path=failing_path, refused_path=refused_path)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(CLI_PATH), *args], capture_output=True, text=True, timeout=60)


def calls_of(fake_tools: dict) -> list[str]:
    return fake_tools["calls_path"].read_text().splitlines()


class TestInstall:
    def test_it_checks_then_vendors_then_upgrades(self, fake_tools):
        """The three steps a user would otherwise run by hand, in the only order that works."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--image-tag", "v1")
        calls = fake_tools["calls_path"].read_text().splitlines()

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("kubectl auth can-i") for call in calls)
        assert f"helm dependency build {CHART_DIR}" in calls
        assert f"helm upgrade --install wb {CHART_DIR} -n rl --set-string infra.image.tag=v1" in calls
        assert calls.index(f"helm dependency build {CHART_DIR}") < calls.index(
            f"helm upgrade --install wb {CHART_DIR} -n rl --set-string infra.image.tag=v1"
        )

    def test_a_failed_doctor_stops_before_helm_runs(self, fake_tools):
        """Installing anyway would leave a workbench whose token grants nothing."""
        (fake_tools["failing_path"]).write_text("kubectl")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 1
        assert "helm upgrade" not in fake_tools["calls_path"].read_text()

    def test_the_rbac_options_reach_both_the_doctor_and_helm(self, fake_tools):
        """Passing them to only one of the two is the mismatch this command exists to remove."""
        run_cli("install", "-n", "rl", "-r", "wb", "--no-rbac", "--no-lws")
        calls = fake_tools["calls_path"].read_text()

        assert "auth can-i create serviceaccounts" not in calls
        assert "leaderworkersets" not in calls
        assert "--set rbac.create=false --set rbac.leaderWorkerSets=false" in calls

    def test_values_files_and_raw_overrides_are_passed_through(self, fake_tools, tmp_path):
        """A per-cluster values file plus the occasional --set is how this chart is actually installed."""
        values = tmp_path / "cluster.yaml"
        values.write_text("{}")

        run_cli(
            "install",
            "-n",
            "rl",
            "-r",
            "wb",
            "-f",
            str(values),
            "--set",
            "resources.requests.cpu=4",
        )

        assert f"-f {values} --set resources.requests.cpu=4" in fake_tools["calls_path"].read_text()

    def test_the_doctor_can_be_skipped(self, fake_tools):
        """A denied check is sometimes a wrong answer from the cluster, not a real blocker."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "auth can-i" not in fake_tools["calls_path"].read_text()

    def test_an_existing_namespace_is_left_alone(self, fake_tools):
        """Creating a namespace that already exists needs cluster-scoped rights this user is not assumed to have."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl get namespace -- rl" in calls_of(fake_tools)
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_missing_namespace_is_created_before_anything_else(self, fake_tools):
        """helm cannot install into a namespace that does not exist, and the doctor would fail on it first."""
        fake_tools["refused_path"].write_text("NotFound|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" in calls
        assert calls.index("kubectl create namespace rl") < min(
            index for index, call in enumerate(calls) if call.startswith("helm")
        )

    def test_a_namespace_this_account_may_not_read_is_not_created(self, fake_tools):
        """A namespace-scoped user always gets Forbidden here, and creating one needs rights they never have."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)
        assert "kubectl get serviceaccounts -n rl -o name" in calls_of(fake_tools)

    def test_a_namespace_absent_from_inside_stops_the_install(self, fake_tools):
        """Every later step would fail on the missing namespace, one confusing error at a time."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace\nNotFound|kubectl get serviceaccounts")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 1
        assert "does not exist" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_namespace_lookup_that_fails_for_another_reason_is_not_a_missing_namespace(self, fake_tools):
        """A timed-out apiserver would otherwise send the user into a create that fails for a third reason."""
        fake_tools["failing_path"].write_text("kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 1
        assert "could not read namespace rl" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_the_doctor_runs_before_helm_touches_the_cluster(self, fake_tools):
        """Checking after the install would report on a cluster the install has already changed."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        first_can_i = min(index for index, call in enumerate(calls) if "auth can-i" in call)
        first_upgrade = min(index for index, call in enumerate(calls) if call.startswith("helm upgrade"))
        assert first_can_i < first_upgrade


class TestNamespaceOccupancy:
    def test_the_check_spans_the_kinds_that_kubectl_get_all_leaves_out(self, fake_tools):
        """The Role covers configmaps, secrets and RBAC too, none of which `kubectl get all` ever returns."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        for kind in ("configmap", "secret", "persistentvolumeclaim", "serviceaccount"):
            assert f"kubectl get {kind} -n rl -l app.kubernetes.io/managed-by!=Helm -o name" in calls

    def test_a_kind_this_account_may_not_list_fails_instead_of_passing_quietly(self, fake_tools):
        """Unreadable is not empty, and a silent pass hands the workbench a namespace nobody has inspected."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get secret")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 1
        assert "could not list secret" in result.stderr
        assert "helm upgrade" not in fake_tools["calls_path"].read_text()


class TestDryRun:
    def test_it_runs_the_checks_without_installing(self, fake_tools):
        """The same checks install runs, for someone who wants to know before touching the cluster."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
        calls = fake_tools["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl auth can-i" in calls
        assert "helm upgrade" not in calls

    def test_it_fails_when_a_check_fails(self, fake_tools):
        """Its exit code is what a wrapper script would branch on."""
        (fake_tools["failing_path"]).write_text("kubectl")

        assert run_cli("install", "--dry-run", "-n", "rl", "-r", "wb").returncode == 1

    def test_it_changes_nothing_in_the_cluster(self, fake_tools):
        """A dry run that created the namespace or rolled out the chart would defeat the point of asking first."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb", "--image-tag", "v1")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert not any(call.startswith("kubectl create namespace") for call in calls)
        assert not any(call.startswith("helm upgrade") for call in calls)
        assert not any("rollout status" in call for call in calls)

    def test_it_changes_nothing_on_the_local_disk_either(self, fake_tools):
        """Vendoring dependencies into the checkout is a write, and a dry run promises not to write."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
        builds = [call for call in calls_of(fake_tools) if call.startswith("helm dependency build")]

        assert result.returncode == 0, result.stdout + result.stderr
        assert builds, "the dry run still has to render the chart to check what it grants"
        assert not any(call.endswith(str(CHART_DIR)) for call in builds)

    def test_it_still_renders_the_chart_it_checks(self, fake_tools):
        """A dry run that skipped the render would stop reporting the rbac the chart asks for."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("helm template wb ") for call in calls_of(fake_tools))


class TestExec:
    def test_it_execs_into_the_statefulset_of_that_release(self, fake_tools):
        """The pod name is derived from the release, which is the only thing the user knows."""
        run_cli("exec", "-n", "rl", "-r", "miles-workbench-alice")

        assert "kubectl exec -it statefulset/miles-workbench-alice -n rl -- bash" in (
            fake_tools["calls_path"].read_text()
        )

    def test_the_kubectl_separator_is_not_passed_twice(self, fake_tools):
        """Typing `--` before the command is kubectl muscle memory; a second one becomes the executable."""
        run_cli("exec", "-n", "rl", "-r", "wb", "--", "echo", "hi")

        assert "kubectl exec -it statefulset/wb-miles-workbench -n rl -- echo hi" in (
            fake_tools["calls_path"].read_text()
        )

    def test_a_command_can_be_given_instead_of_a_shell(self, fake_tools):
        """Scripting against the workbench should not need a second tool."""
        run_cli("exec", "-n", "rl", "-r", "wb", "python", "-c", "print(1)")

        assert "kubectl exec -it statefulset/wb-miles-workbench -n rl -- python -c print(1)" in (
            fake_tools["calls_path"].read_text()
        )


class TestMissingBinaries:
    @pytest.mark.parametrize(
        "args", [["exec", "-n", "rl", "-r", "wb"], ["install", "-n", "rl", "-r", "wb", "--skip-doctor"]]
    )
    def test_a_missing_binary_is_reported_not_raised(self, tmp_path, monkeypatch, args):
        """It runs before any Miles environment exists, so an absent client is the expected first failure."""
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
        result = run_cli(*args)

        assert result.returncode == 1
        assert "is installed" in result.stderr
        assert "Traceback" not in result.stderr


class TestCli:
    @pytest.mark.parametrize(
        "args",
        [
            [],
            ["install"],
            ["install", "-n", "rl"],
            ["exec", "-n", "rl"],
            ["install", "--dry-run", "-r", "wb"],
            ["bogus"],
        ],
    )
    def test_incomplete_invocations_are_usage_errors(self, fake_tools, args):
        """Half a command must not reach helm or kubectl."""
        result = run_cli(*args)

        assert result.returncode == 2
        assert not fake_tools["calls_path"].exists()

    def test_it_is_executable_and_runs_on_the_python_a_laptop_ships_with(self):
        """It runs before any miles environment exists, so it must not need a managed interpreter."""
        assert os.access(CLI_PATH, os.X_OK)
        result = subprocess.run(["/usr/bin/python3", str(CLI_PATH), "--help"], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr

    def test_it_needs_nothing_but_the_standard_library(self):
        """It is the first thing a new user runs, before any miles environment exists."""
        imports = {
            line.split()[1].split(".")[0]
            for line in CLI_PATH.read_text().splitlines()
            if line.startswith("import ") or line.startswith("from ")
        }

        assert imports <= set(sys.stdlib_module_names) | {"__future__"}
