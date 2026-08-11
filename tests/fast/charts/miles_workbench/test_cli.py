import subprocess
import sys
from pathlib import Path

import pytest
from tests.fast.charts.utils import CHART_DIR, REPO_ROOT, run_workbench

from miles.utils.external_utils.miles_workbench import __main__ as workbench_main
from miles.utils.external_utils.miles_workbench.naming import DEFAULT_RELEASE, PACKAGE, object_name, run_release_name


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "calls.log"
    failing_path = tmp_path / "failing"
    failing_path.write_text("")
    refused_path = tmp_path / "refused"
    refused_path.write_text("")
    pods_path = tmp_path / "pods"
    pods_path.write_text("")

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
            f'if [ "{binary} $1 $2" = "kubectl get pods" ]; then\n'
            '  items=""\n'
            '  while IFS= read -r pod || [ -n "$pod" ]; do\n'
            '    [ -z "$pod" ] && continue\n'
            '    [ -n "$items" ] && items="$items,"\n'
            '    items="$items{\\"metadata\\":{\\"name\\":\\"$pod\\",\\"uid\\":\\"$pod\\"}}"\n'
            f"  done < {pods_path}\n"
            '  [ -n "$items" ] && echo "{\\"items\\":[$items]}"\n'
            "fi\n"
            "exit 0\n"
        )
        (bin_dir / binary).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    return dict(calls_path=calls_path, failing_path=failing_path, refused_path=refused_path, pods_path=pods_path)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return run_workbench(*args, capture_output=True)


def calls_of(fake_tools: dict) -> list[str]:
    return fake_tools["calls_path"].read_text().splitlines()


class TestInstall:
    def test_it_checks_then_vendors_then_upgrades(self, fake_tools):
        """The three steps a user would otherwise run by hand, in the only order that works."""
        result = run_cli("install", "-n", "rl", "-r", "wb")
        calls = fake_tools["calls_path"].read_text().splitlines()

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("kubectl auth can-i") for call in calls)
        assert f"helm dependency build {CHART_DIR}" in calls
        assert (
            f"helm upgrade --install wb {CHART_DIR} --namespace rl --set-string objectName=wb-miles-workbench" in calls
        )
        assert calls.index(f"helm dependency build {CHART_DIR}") < calls.index(
            f"helm upgrade --install wb {CHART_DIR} --namespace rl --set-string objectName=wb-miles-workbench"
        )

    def test_a_failed_preflight_stops_before_helm_runs(self, fake_tools):
        """Installing anyway would leave a workbench whose token grants nothing."""
        (fake_tools["failing_path"]).write_text("kubectl")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 1
        assert "helm upgrade" not in fake_tools["calls_path"].read_text()

    def test_the_rbac_options_reach_both_the_preflight_and_helm(self, fake_tools):
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

    def test_the_preflight_can_be_skipped(self, fake_tools):
        """A denied check is sometimes a wrong answer from the cluster, not a real blocker."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "auth can-i" not in fake_tools["calls_path"].read_text()

    def test_an_existing_namespace_is_left_alone(self, fake_tools):
        """Creating a namespace that already exists needs cluster-scoped rights this user is not assumed to have."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl get namespace -- rl" in calls_of(fake_tools)
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_missing_namespace_is_created_before_anything_else(self, fake_tools):
        """helm cannot install into a namespace that does not exist, and the preflight would fail on it first."""
        fake_tools["refused_path"].write_text("NotFound|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" in calls
        assert calls.index("kubectl create namespace rl") < min(
            index for index, call in enumerate(calls) if call.startswith("helm")
        )

    def test_a_namespace_this_account_may_not_read_is_not_created(self, fake_tools):
        """A namespace-scoped user always gets Forbidden here, and creating one needs rights they never have."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)
        assert "kubectl get serviceaccounts -n rl -o name" in calls_of(fake_tools)

    def test_a_namespace_absent_from_inside_stops_the_install(self, fake_tools):
        """Every later step would fail on the missing namespace, one confusing error at a time."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace\nNotFound|kubectl get serviceaccounts")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 1
        assert "does not exist" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_namespace_lookup_that_fails_for_another_reason_is_not_a_missing_namespace(self, fake_tools):
        """A timed-out apiserver would otherwise send the user into a create that fails for a third reason."""
        fake_tools["failing_path"].write_text("kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 1
        assert "could not read namespace rl" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_the_preflight_runs_before_helm_touches_the_cluster(self, fake_tools):
        """Checking after the install would report on a cluster the install has already changed."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        first_can_i = min(index for index, call in enumerate(calls) if "auth can-i" in call)
        first_upgrade = min(index for index, call in enumerate(calls) if call.startswith("helm upgrade"))
        assert first_can_i < first_upgrade

    def test_it_waits_for_the_pod_with_a_bounded_timeout(self, fake_tools):
        """An unbounded wait hangs a CI job forever when the image cannot be pulled."""
        run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight", "--timeout", "42")

        assert "kubectl rollout status statefulset/wb-miles-workbench -n rl --timeout=42s" in calls_of(fake_tools)

    def test_the_wait_has_a_timeout_even_when_none_is_asked_for(self, fake_tools):
        """The default matters more than the flag: almost nobody passes --timeout."""
        run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert "kubectl rollout status statefulset/wb-miles-workbench -n rl --timeout=600s" in calls_of(fake_tools)

    def test_a_pod_that_never_becomes_ready_fails_and_shows_why(self, fake_tools):
        """Exiting zero on a pending pod sends the user to exec into a pod that is not there."""
        fake_tools["failing_path"].write_text("kubectl rollout status")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")
        calls = calls_of(fake_tools)

        assert result.returncode != 0
        assert "was not ready within 600s" in result.stderr
        assert "kubectl get pods -n rl" in calls
        assert "kubectl describe statefulset wb-miles-workbench -n rl" in calls

    def test_a_successful_install_prints_the_command_that_gets_a_shell(self, fake_tools):
        """The install is only half the workflow; the next step must not need the README."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert f"python -m {PACKAGE} exec -n rl -r wb" in result.stderr


class TestNamespaceOccupancy:
    def test_the_check_spans_the_kinds_that_kubectl_get_all_leaves_out(self, fake_tools):
        """The Role covers configmaps, secrets and RBAC too, none of which `kubectl get all` ever returns."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        for kind in ("configmap", "secret", "persistentvolumeclaim", "serviceaccount"):
            assert f"kubectl get {kind} -n rl -l app.kubernetes.io/managed-by!=Helm -o name" in calls

    def test_a_kind_this_account_may_not_list_is_said_out_loud(self, fake_tools):
        """Unreadable is not empty, and a silent pass hands the workbench a namespace nobody has inspected."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get secret")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "could not check whether namespace rl holds nothing but Miles releases" in result.stderr
        assert "listing secret failed" in result.stderr


class TestUninstall:
    def test_it_removes_the_release_and_keeps_the_namespace(self, fake_tools):
        """The namespace usually predates the release and may hold a colleague's work."""
        result = run_cli("uninstall", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "helm uninstall wb --namespace rl" in calls
        assert not any("delete namespace" in call for call in calls)


class TestStop:
    def test_it_names_the_release_a_run_id_stands_for(self, fake_tools):
        """A user knows the run id they launched with, not the release name the launcher derived."""
        run_cli("stop", "-n", "rl", "demo")

        assert f"helm uninstall {run_release_name('demo')} --namespace rl" in calls_of(fake_tools)


class TestCollectDiagnosis:
    def test_it_writes_one_directory_holding_logs_describes_and_events(self, fake_tools, tmp_path):
        """A support request is one directory to attach, not a list of kubectl commands to re-run by hand."""
        fake_tools["pods_path"].write_text("wb-0\norchestrator-0\n")
        output_dir = tmp_path / "out"

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(output_dir))
        written = Path(result.stdout.strip())

        assert result.returncode == 0, result.stdout + result.stderr
        assert written.parent == output_dir
        assert [path.name for path in output_dir.iterdir()] == [written.name]
        assert {path.name for path in written.iterdir()} == {
            "events.txt",
            "wb-0.log",
            "wb-0.previous.log",
            "wb-0.describe.txt",
            "orchestrator-0.log",
            "orchestrator-0.previous.log",
            "orchestrator-0.describe.txt",
        }

    def test_a_container_that_never_restarted_leaves_no_empty_previous_log(self, fake_tools, tmp_path):
        """kubectl fails on --previous for a first-boot container, and an empty file reads as a lost log."""
        fake_tools["pods_path"].write_text("wb-0\n")
        fake_tools["failing_path"].write_text("kubectl logs wb-0 -n rl --all-containers --previous")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode == 0, result.stdout + result.stderr
        assert (written / "wb-0.previous.log").exists() is False
        assert (written / "wb-0.log").exists()

    def test_a_restarted_container_keeps_the_log_that_explains_the_restart(self, fake_tools, tmp_path):
        """The crash is in the previous container's log, not in the one the running container is writing."""
        fake_tools["pods_path"].write_text("wb-0\n")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))

        assert (Path(result.stdout.strip()) / "wb-0.previous.log").exists()

    def test_the_run_verdict_is_left_out_when_no_run_directory_is_named(self, fake_tools, tmp_path):
        """Most diagnoses are of the workbench itself, where no orchestrator has ever written a verdict."""
        fake_tools["pods_path"].write_text("wb-0\n")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))

        assert (Path(result.stdout.strip()) / "orchestrator.state").exists() is False

    def test_a_named_run_directory_contributes_the_verdict_of_its_latest_launch(self, fake_tools, tmp_path):
        """A relaunched run keeps every launch's exit file, and only the newest is this run's verdict."""
        fake_tools["pods_path"].write_text("wb-0\n")
        state = tmp_path / "run" / "state"
        state.mkdir(parents=True)
        (state / "orchestrator-260101-000000-000001.state").write_text("0\n")
        (state / "orchestrator-260101-010000-000002.state").write_text("1\n")

        result = run_cli(
            "collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path), "--run-dir", str(tmp_path / "run")
        )

        collected = Path(result.stdout.strip()) / "orchestrator-260101-010000-000002.state"
        assert collected.read_text() == "1\n"

    def test_a_run_directory_with_no_verdict_yet_is_reported_as_incomplete(self, fake_tools, tmp_path):
        """Asking for a run's verdict and silently getting none would read as the run having succeeded."""
        fake_tools["pods_path"].write_text("wb-0\n")
        (tmp_path / "run" / "state").mkdir(parents=True)

        result = run_cli(
            "collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path), "--run-dir", str(tmp_path / "run")
        )

        assert result.returncode == 1
        assert "a verdict under" in result.stderr

    def test_an_unreadable_cluster_still_produces_a_directory(self, fake_tools, tmp_path):
        """It is run when things are broken, so a failed pod listing must not cost the events and the verdict."""
        fake_tools["failing_path"].write_text("kubectl get pods")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode != 0
        assert "pod listing in namespace rl" in result.stderr
        assert "pod listing in namespace rl" in result.stderr
        assert {path.name for path in written.iterdir()} == {"events.txt"}

    def test_pod_logs_cover_every_container_in_the_pod(self, fake_tools, tmp_path):
        """The orchestrator pod runs sidecars, and the default single-container log would silently drop them."""
        fake_tools["pods_path"].write_text("wb-0\n")

        run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        calls = calls_of(fake_tools)

        assert "kubectl logs wb-0 -n rl --all-containers" in calls
        assert "kubectl logs wb-0 -n rl --all-containers --previous" in calls

    def test_a_failed_collection_step_is_reported_and_fails_the_run(self, fake_tools, tmp_path):
        """A caller that branches on the exit code must not read a half-collected directory as a whole one."""
        fake_tools["pods_path"].write_text("wb-0\n")
        fake_tools["failing_path"].write_text("kubectl logs wb-0 -n rl --all-containers\nkubectl get events")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode != 0
        assert "logs of wb-0" in result.stderr
        assert "events" in result.stderr
        assert {path.name for path in written.iterdir()} == {"events.txt", "wb-0.log", "wb-0.describe.txt"}


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
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
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

    def test_it_reaches_the_default_release_when_none_is_named(self, fake_tools):
        """Most namespaces hold one workbench, and naming it in every command is noise."""
        run_cli("exec", "-n", "rl")

        assert f"kubectl exec -it statefulset/{object_name(DEFAULT_RELEASE)} -n rl -- bash" in (
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
        "args", [["exec", "-n", "rl", "-r", "wb"], ["install", "-n", "rl", "-r", "wb", "--skip-preflight"]]
    )
    def test_a_missing_binary_is_reported_not_raised(self, tmp_path, monkeypatch, args):
        """It runs before any Miles environment exists, so an absent client is the expected first failure."""
        # an empty PATH, not one that merely omits a directory: a runner that ships kubectl in
        # /usr/bin would run it for real and fail on the cluster instead of on the missing client
        monkeypatch.setenv("PATH", str(tmp_path))
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
            ["install", "--dry-run", "-r", "wb"],
            ["bogus"],
        ],
    )
    def test_incomplete_invocations_are_usage_errors(self, fake_tools, args):
        """Half a command must not reach helm or kubectl."""
        result = run_cli(*args)

        assert result.returncode == 2
        assert not fake_tools["calls_path"].exists()

    def test_the_object_name_leaves_room_for_the_names_kubernetes_derives_from_it(self):
        """Pod hostnames and the controller-revision-hash label are 63-char names built from this one."""
        name = object_name("a" * 53)

        assert len(f"{name}-0") <= 63
        assert len(name) + len("-0123456789") <= 63

    def test_the_object_name_carries_the_chart_name_only_once(self):
        """A release a user already named after the chart must not become miles-workbench-miles-workbench."""
        assert object_name("miles-workbench-alice") == "miles-workbench-alice"
        assert object_name("alice") == "alice-miles-workbench"

    def test_it_pulls_in_no_training_dependencies(self):
        """It installs the pod that holds the training environment, so it cannot need that environment itself."""
        probe = f"import importlib, sys; importlib.import_module('{PACKAGE}.__main__'); print(*sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
        )

        assert result.returncode == 0, result.stderr
        assert {"torch", "ray", "megatron", "transformers"}.isdisjoint(result.stdout.split())

    def test_it_offers_exactly_these_subcommands(self):
        """The subcommand set is the cli's whole contract with a runbook, so a rename must be deliberate."""
        assert {command.name for command in workbench_main.app.registered_commands} == {
            "install",
            "exec",
            "stop",
            "uninstall",
            "collect-diagnosis",
        }
