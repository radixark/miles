import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.fast.charts.utils import (
    CHART_DIR,
    CLI_PATH,
    LWS_RESOURCE,
    NAMESPACE,
    can_i_queries,
    cli_module,
    render,
    requires_helm,
    single_object_of_kind,
)


ROLES = "roles.rbac.authorization.k8s.io"
NO_DELEGATION = " ".join(
    f"{verb}:{ROLES}{suffix}" for verb in ("escalate", "bind") for suffix in ("", "/miles-workbench-alice")
)


class TestDoctor:
    @pytest.fixture
    def fake_cluster(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calls_path = tmp_path / "calls.log"
        denied_path = tmp_path / "denied"
        denied_path.write_text("")
        absent_path = tmp_path / "absent"
        absent_path.write_text("")
        unreachable_path = tmp_path / "unreachable"
        unreachable_path.write_text("")
        broken_path = tmp_path / "broken"
        broken_path.write_text("")
        forbidden_path = tmp_path / "forbidden"
        forbidden_path.write_text("")
        deny_all_path = tmp_path / "deny-all"
        deny_all_path.write_text("")

        (bin_dir / "kubectl").write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$@" >> {calls_path}\n'
            "positional=0\n"
            "skip=0\n"
            'for arg in "${@:3}"; do\n'
            '  if [ "$skip" -eq 1 ]; then skip=0; continue; fi\n'
            '  case "$arg" in\n'
            "    -n|--namespace) skip=1 ;;\n"
            "    -*) ;;\n"
            "    *) positional=$((positional + 1)) ;;\n"
            "  esac\n"
            "done\n"
            'if [ "$1 $2" = "auth can-i" ] && [ "$positional" -gt 2 ]; then\n'
            '  echo "error: you must specify two arguments: verb resource or verb resource/resourceName." >&2\n'
            "  exit 1\n"
            "fi\n"
            'if [ "$1 $2" = "auth can-i" ]; then\n'
            f"  [ -s {deny_all_path} ] && exit 1\n"
            f"  for denied in $(cat {denied_path}); do\n"
            '    case "$denied" in\n'
            '      *:*) [ "$3" = "${denied%%:*}" ] && [ "$4" = "${denied#*:}" ] && exit 1 ;;\n'
            '      *) for arg in "$@"; do [ "$arg" = "$denied" ] && exit 1; done ;;\n'
            "    esac\n"
            "  done\n"
            "fi\n"
            'if [ "$1" = "get" ] && [ "$2" = "--raw" ]; then\n'
            f"  if [ -s {unreachable_path} ]; then\n"
            f"    cat {unreachable_path} >&2\n"
            "    exit 1\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "get" ]; then\n'
            "  terminated=0\n"
            '  for arg in "${@:3}"; do\n'
            '    if [ "$arg" = "--" ]; then terminated=1; continue; fi\n'
            '    if [ "$terminated" -eq 0 ]; then\n'
            '      case "$arg" in -*)\n'
            '        echo "error: unknown shorthand flag in $arg" >&2\n'
            "        exit 1 ;;\n"
            "      esac\n"
            "    fi\n"
            "  done\n"
            '  name="${@: -1}"\n'
            f"  for absent in $(cat {absent_path}); do\n"
            '    if [ "$name" = "$absent" ]; then\n'
            '      echo "Error from server (NotFound): $2 \\"$name\\" not found" >&2\n'
            "      exit 1\n"
            "    fi\n"
            "  done\n"
            f"  for forbidden in $(cat {forbidden_path}); do\n"
            '    if [ "$name" = "$forbidden" ]; then\n'
            '      echo "Error from server (Forbidden): $2 is forbidden" >&2\n'
            "      exit 1\n"
            "    fi\n"
            "  done\n"
            f"  for broken in $(cat {broken_path}); do\n"
            '    if [ "$name" = "$broken" ]; then\n'
            '      echo "Unable to connect to the server: dial tcp: i/o timeout" >&2\n'
            "      exit 1\n"
            "    fi\n"
            "  done\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        (bin_dir / "helm").write_text("#!/usr/bin/env bash\nexit 0\n")
        for binary in ("kubectl", "helm"):
            (bin_dir / binary).chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

        return dict(
            bin_dir=bin_dir,
            calls_path=calls_path,
            denied_path=denied_path,
            absent_path=absent_path,
            unreachable_path=unreachable_path,
            broken_path=broken_path,
            forbidden_path=forbidden_path,
            deny_all_path=deny_all_path,
        )

    def run_doctor(self, *args: str) -> subprocess.CompletedProcess:
        release = [] if {"-r", "--release"} & set(args) else ["-r", "miles-workbench-alice"]
        return subprocess.run([str(CLI_PATH), "doctor", *release, *args], capture_output=True, text=True, timeout=60)

    def test_it_asks_for_exactly_the_rights_the_install_and_the_workflow_need(self, fake_cluster):
        """Pinned in full: a missing check is a false pass, and a spurious one turns away a legitimate installer."""
        cli = cli_module()
        expected = can_i_queries(
            {
                **cli.WORKFLOW_RULES,
                **cli.CHART_RULES,
                **cli.CHART_SERVICE_ACCOUNT_RULES,
                **cli.CHART_RBAC_RULES,
                **cli.GRANTED_RULES,
                **cli.GRANTED_LWS_RULES,
            }
        )

        result = self.run_doctor("-n", "rl")
        calls = fake_cluster["calls_path"].read_text().splitlines()
        queries = {
            line.removeprefix("auth can-i ").removesuffix(" -n rl") for line in calls if line.startswith("auth")
        }

        assert result.returncode == 0, result.stdout + result.stderr
        assert "doctor passed" in result.stdout
        assert queries == expected
        assert "get namespace -- rl" in calls
        assert f"get crd -- {LWS_RESOURCE}" in calls

    @requires_helm
    @pytest.mark.parametrize("release", ["miles-workbench-alice", "wb", "a" * 53])
    def test_the_role_name_it_computes_is_the_one_the_chart_renders(self, release):
        """It asks about a Role by name, so its naming must not drift from the chart's own helper."""
        result = subprocess.run(
            ["helm", "template", release, str(CHART_DIR), "-n", NAMESPACE],
            capture_output=True,
            text=True,
        )
        objects = [document for document in yaml.safe_load_all(result.stdout) if document is not None]
        rendered = single_object_of_kind(objects, "Role")["metadata"]["name"]

        assert cli_module().role_name(release) == rendered

    @requires_helm
    def test_the_checked_grants_are_exactly_the_role_the_chart_ships(self):
        """The doctor exists to predict the install, so its rule table cannot drift from the rendered Role."""
        rendered = {}
        for rule in single_object_of_kind(render(), "Role")["rules"]:
            for group in rule["apiGroups"]:
                for resource in rule["resources"]:
                    name, _, subresource = resource.partition("/")
                    key = name if group == "" else f"{name}.{group}"
                    assert "resourceNames" not in rule, "a name-restricted rule needs a name-qualified check"
                    rendered[f"{key}/{subresource}" if subresource else key] = set(rule["verbs"])
        cli = cli_module()
        checked = {resource: set(verbs) for resource, verbs in {**cli.GRANTED_RULES, **cli.GRANTED_LWS_RULES}.items()}

        assert rendered == checked

    def test_a_denied_rule_fails_the_run(self, fake_cluster):
        """Preflight exists to catch missing rights before install, so a denial must be a hard failure."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "configmaps" in result.stderr

    def test_escalate_and_bind_together_downgrade_a_missing_grant_to_a_warning(self, fake_cluster):
        """Those are the two verbs Kubernetes checks: escalate for the Role, bind for its RoleBinding."""
        (fake_cluster["denied_path"]).write_text("configmaps")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARN" in result.stderr

    @pytest.mark.parametrize("verb", ["bind", "escalate"])
    def test_a_right_restricted_to_this_release_counts(self, fake_cluster, verb):
        """helm applies the Role server-side, so Kubernetes checks both verbs against the object's name."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {verb}:roles.rbac.authorization.k8s.io")

        result = self.run_doctor("-n", "rl", "-r", "miles-workbench-alice")

        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            f"auth can-i {verb} roles.rbac.authorization.k8s.io/miles-workbench-alice -n rl"
            in fake_cluster["calls_path"].read_text()
        )

    def test_escalate_without_bind_is_still_a_failure(self, fake_cluster):
        """The Role would be created and its RoleBinding refused, leaving a token that grants nothing."""
        (fake_cluster["denied_path"]).write_text(
            "configmaps bind:roles.rbac.authorization.k8s.io"
            " bind:roles.rbac.authorization.k8s.io/miles-workbench-alice"
        )

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "may grant the workbench its Role" in result.stderr

    def test_a_grant_denied_without_escalate_is_a_failure(self, fake_cluster):
        """Without escalate the Role can only carry rules the installer holds, so the install would be rejected."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "may grant the workbench its Role" in result.stderr

    def test_missing_leaderworkerset_rights_name_the_admin_prerequisite(self, fake_cluster):
        """LWS rights come from a cluster admin, so that denial must point there rather than at the user."""
        (fake_cluster["denied_path"]).write_text(f"{LWS_RESOURCE} {NO_DELEGATION}")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "cluster admin" in result.stderr

    def test_an_unrelated_denial_does_not_blame_the_lws_prerequisite(self, fake_cluster):
        """Sending a user to their cluster admin over their own missing configmap rights wastes everyone's time."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "cluster admin" not in result.stderr

    def test_an_rbac_free_install_checks_only_what_it_installs(self, fake_cluster):
        """With rbac.create=false an admin pre-creates the identity, so none of that is the installer's to do."""
        (fake_cluster["denied_path"]).write_text("roles.rbac.authorization.k8s.io serviceaccounts configmaps")

        result = self.run_doctor("-n", "rl", "--no-rbac")
        calls = fake_cluster["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "roles.rbac.authorization.k8s.io" not in calls
        assert "serviceaccounts" not in calls
        assert "configmaps" not in calls

    def test_turning_off_leaderworkersets_drops_those_checks(self, fake_cluster):
        """A cluster without LWS installed cannot grant those rights, and must still get a workbench."""
        (fake_cluster["denied_path"]).write_text(LWS_RESOURCE)

        result = self.run_doctor("-n", "rl", "--no-lws")
        calls = fake_cluster["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "leaderworkersets" not in calls

    def test_a_missing_namespace_fails(self, fake_cluster):
        """A mistyped -n is the likeliest user error, and every can-i answer would still look fine."""
        (fake_cluster["absent_path"]).write_text("rl")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "namespace rl exists" in result.stderr

    def test_a_missing_lws_crd_fails(self, fake_cluster):
        """Without the CRD the Role would grant rights over a resource that does not exist."""
        (fake_cluster["absent_path"]).write_text(LWS_RESOURCE)

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert f"crd {LWS_RESOURCE} exists" in result.stderr

    def test_a_lookup_that_is_merely_forbidden_is_ignored(self, fake_cluster):
        """A namespace-scoped user cannot read cluster-scoped objects, which is not a reason to fail."""
        (fake_cluster["forbidden_path"]).write_text(f"rl {LWS_RESOURCE}")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_lookup_that_fails_for_any_other_reason_is_not_a_pass(self, fake_cluster):
        """Treating an unreadable answer as "present" is how a transient outage certifies a missing object."""
        (fake_cluster["broken_path"]).write_text("rl")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "namespace rl exists" in result.stderr

    def test_a_missing_object_whose_name_contains_forbidden_still_fails(self, fake_cluster):
        """The Forbidden test must key off the server's status, not off text that a name can contain."""
        (fake_cluster["absent_path"]).write_text("forbidden-ns")

        result = self.run_doctor("--namespace=forbidden-ns")

        assert result.returncode == 1
        assert "namespace forbidden-ns exists" in result.stderr

    def test_names_are_looked_up_after_the_option_terminator(self, fake_cluster):
        """kubectl would read a dash-leading name as flags, so lookups must pass it after "--"."""
        self.run_doctor("--namespace=-dashed")

        assert "get namespace -- -dashed" in fake_cluster["calls_path"].read_text()

    @pytest.mark.parametrize(
        "error",
        [
            "The connection to the server localhost:8080 was refused",
            "error: exec plugin: invalid apiVersion client.authentication",
            "error: You must be logged in to the server (Unauthorized)",
        ],
    )
    def test_any_failure_to_reach_the_server_stops_the_run(self, fake_cluster, error):
        """/version is served to every authenticated caller, so a failure here is never a permission problem."""
        (fake_cluster["unreachable_path"]).write_text(error)

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "cluster is reachable" in result.stderr
        assert "kubeconfig" in result.stderr
        assert "auth can-i" not in fake_cluster["calls_path"].read_text()

    def test_a_wholly_denied_run_points_at_the_namespace_and_context(self, fake_cluster):
        """A mistyped namespace denies everything, and a namespace-scoped user cannot look the namespace up."""
        (fake_cluster["deny_all_path"]).write_text("1")

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "confirm the namespace name and your kubectl context" in result.stderr

    def test_missing_binaries_are_reported_before_any_query(self, fake_cluster):
        """Without helm there is nothing to check, and the message must say which binary is missing."""
        (fake_cluster["bin_dir"] / "helm").unlink()

        result = self.run_doctor("-n", "rl")

        assert result.returncode == 1
        assert "helm is installed" in result.stderr
        assert fake_cluster["calls_path"].exists() is False

    def test_the_namespace_is_required(self, fake_cluster):
        """Every check is namespace-scoped, so a missing namespace is a usage error, not a failed check."""
        result = self.run_doctor()

        assert result.returncode == 2
        assert "--namespace" in result.stderr

    @pytest.mark.parametrize(
        "args", [["-n"], ["--namespace"], ["--namespace="], ["-n", "rl", "-r"], ["-n", "rl", "--release="]]
    )
    def test_a_flag_without_its_value_is_a_usage_error(self, fake_cluster, args):
        """A dangling flag must fail loudly instead of consuming the next flag or spinning in the parser."""
        result = self.run_doctor(*args)

        assert result.returncode == 2

    def test_an_unknown_argument_is_a_usage_error(self, fake_cluster):
        """A mistyped flag must not silently degrade into a partial check."""
        result = self.run_doctor("-n", "rl", "--deploy")

        assert result.returncode == 2
        assert "--deploy" in result.stderr

    def test_asking_for_help_succeeds(self, fake_cluster):
        """--help is not an error, and scripts wrapping the doctor read its exit code."""
        result = self.run_doctor("--help")

        assert result.returncode == 0
        assert "usage: cli.py doctor" in result.stdout

    def test_captured_output_keeps_the_checks_in_order(self, fake_cluster):
        """Users redirect this into a ticket; block-buffered stdout would float the failures to the top."""
        result = subprocess.run(
            [str(CLI_PATH), "doctor", "-n", "rl", "-r", "wb"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )

        assert result.stdout.splitlines()[0] == "PASS  kubectl is installed"

    def test_the_script_is_executable_and_needs_nothing_but_the_standard_library(self):
        """Users run it from a clone before any miles install exists, so it must be self-contained."""
        assert os.access(CLI_PATH, os.X_OK)
        assert (
            subprocess.run([sys.executable, "-c", f"compile(open({str(CLI_PATH)!r}).read(), 'p', 'exec')"]).returncode
            == 0
        )
        imports = {
            line.split()[1].split(".")[0]
            for line in CLI_PATH.read_text().splitlines()
            if line.startswith("import ") or line.startswith("from ")
        }
        assert imports <= set(sys.stdlib_module_names) | {"__future__"}

    @pytest.mark.skipif(not Path("/usr/bin/python3").exists(), reason="no system python to check against")
    def test_it_runs_on_the_python_a_laptop_ships_with(self):
        """It runs before any miles environment exists, so it must not need a modern or managed interpreter."""
        result = subprocess.run(["/usr/bin/python3", str(CLI_PATH), "--help"], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
