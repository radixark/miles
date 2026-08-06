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

    for binary in ("helm", "kubectl"):
        (bin_dir / binary).write_text(
            "#!/usr/bin/env bash\n"
            f'echo "{binary} $@" >> {calls_path}\n'
            f"for failing in $(cat {failing_path}); do\n"
            f'  [ "{binary}" = "$failing" ] && exit 1\n'
            "done\n"
            "exit 0\n"
        )
        (bin_dir / binary).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    return dict(calls_path=calls_path, failing_path=failing_path)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(CLI_PATH), *args], capture_output=True, text=True, timeout=60)


class TestInstall:
    def test_it_checks_then_vendors_then_upgrades(self, fake_tools):
        """The three steps a user would otherwise run by hand, in the only order that works."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--image-tag", "v1")
        calls = fake_tools["calls_path"].read_text().splitlines()

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("kubectl auth can-i") for call in calls)
        assert f"helm dependency build {CHART_DIR}" in calls
        assert f"helm upgrade --install wb {CHART_DIR} -n rl --set-string image.tag=v1" in calls
        assert calls.index(f"helm dependency build {CHART_DIR}") < calls.index(
            f"helm upgrade --install wb {CHART_DIR} -n rl --set-string image.tag=v1"
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
        (fake_tools["failing_path"]).write_text("kubectl")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-doctor")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "auth can-i" not in fake_tools["calls_path"].read_text()


class TestDoctor:
    def test_it_runs_the_checks_without_installing(self, fake_tools):
        """The same checks install runs, for someone who wants to know before touching the cluster."""
        result = run_cli("doctor", "-n", "rl", "-r", "wb")
        calls = fake_tools["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl auth can-i" in calls
        assert "helm upgrade" not in calls

    def test_it_fails_when_a_check_fails(self, fake_tools):
        """Its exit code is what a wrapper script would branch on."""
        (fake_tools["failing_path"]).write_text("kubectl")

        assert run_cli("doctor", "-n", "rl", "-r", "wb").returncode == 1


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
        [[], ["install"], ["install", "-n", "rl"], ["exec", "-n", "rl"], ["doctor", "-r", "wb"], ["bogus"]],
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
