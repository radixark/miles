import subprocess

import pytest
import yaml
from tests.fast.charts.utils import (
    CHART_DIR,
    LWS_RESOURCE,
    NAMESPACE,
    RELEASE_NAME,
    can_i_queries,
    merged_rules,
    named_object,
    render,
    requires_helm,
    run_workbench,
)

from miles.utils.external_utils.miles_workbench.naming import object_name
from miles.utils.external_utils.miles_workbench.preflight import rules as cli
from miles.utils.external_utils.miles_workbench.render import rbac_plan_of

_FAKE_VERBS = ["get", "list", "watch"]
_FAKE_GRANTED_RULES = {"configmaps": tuple(_FAKE_VERBS), "pods/log": tuple(_FAKE_VERBS)}
_FAKE_GRANTED_LWS_RULES = {cli.LWS_RESOURCE: tuple(_FAKE_VERBS)}


def _fake_role_rules(*, lws: bool) -> list[dict]:
    rules = [{"apiGroups": [""], "resources": ["configmaps", "pods/log"], "verbs": _FAKE_VERBS}]
    if lws:
        rules.append({"apiGroups": [cli.LWS_API_GROUP], "resources": ["leaderworkersets"], "verbs": _FAKE_VERBS})
    return rules


ROLES = "roles.rbac.authorization.k8s.io"
NO_DELEGATION = " ".join(
    f"{verb}:{ROLES}{suffix}" for verb in ("escalate", "bind") for suffix in ("", "/miles-workbench-alice")
)


class TestPreflightChecks:
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
        foreign_path = tmp_path / "foreign"
        foreign_path.write_text("")
        unmanaged_path = tmp_path / "unmanaged"
        unmanaged_path.write_text("")
        family_path = tmp_path / "family"
        family_path.write_text("")
        served_apis_path = tmp_path / "served-apis"
        served_apis_path.write_text(f"{LWS_RESOURCE}\n")
        controller_path = tmp_path / "controller"
        controller_path.write_text("True")

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
            'if [ "$1" = "api-resources" ]; then\n'
            f"  cat {served_apis_path}\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "get" ] && [ "$2" = "--raw" ]; then\n'
            f"  if [ -s {unreachable_path} ]; then\n"
            f"    cat {unreachable_path} >&2\n"
            "    exit 1\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "get" ] && [ "$2" = "all" ]; then\n'
            '  if [[ "$*" == *"managed-by=Helm"* ]]; then\n'
            f"    cat {foreign_path}\n"
            '    if [[ "$*" != *"notin (miles-workbench,miles-run)"* ]]; then\n'
            f"      cat {family_path}\n"
            "    fi\n"
            "  else\n"
            f"    cat {unmanaged_path}\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "get" ]; then\n'
            "  terminated=0\n"
            "  skip=0\n"
            '  for arg in "${@:3}"; do\n'
            '    if [ "$skip" -eq 1 ]; then skip=0; continue; fi\n'
            '    if [ "$arg" = "--" ]; then terminated=1; continue; fi\n'
            '    if [ "$terminated" -eq 0 ]; then\n'
            '      case "$arg" in\n'
            "        -n|--namespace|-l|--selector|-o|--output) skip=1 ;;\n"
            "        -*)\n"
            '          echo "error: unknown shorthand flag in $arg" >&2\n'
            "          exit 1 ;;\n"
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
            f'  if [ "$2" = "deployment.apps" ]; then cat {controller_path}; fi\n'
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        role_path = tmp_path / "role.yaml"
        role_path.write_text(yaml.safe_dump({"kind": "Role", "rules": _fake_role_rules(lws=False)}))
        role_lws_path = tmp_path / "role-lws.yaml"
        role_lws_path.write_text(yaml.safe_dump({"kind": "Role", "rules": _fake_role_rules(lws=True)}))

        (bin_dir / "helm").write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" != "template" ]; then exit 0; fi\n'
            "create=1\n"
            "lws=1\n"
            'for arg in "$@"; do\n'
            '  case "$arg" in\n'
            "    rbac.create=false) create=0 ;;\n"
            "    rbac.create=true) create=1 ;;\n"
            "    rbac.leaderWorkerSets=false) lws=0 ;;\n"
            "    rbac.leaderWorkerSets=true) lws=1 ;;\n"
            "  esac\n"
            "done\n"
            'if [ "$create" = "1" ]; then\n'
            '  echo "---"\n'
            f'  if [ "$lws" = "1" ]; then cat {role_lws_path}; else cat {role_path}; fi\n'
            "fi\n"
            "exit 0\n"
        )
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
            foreign_path=foreign_path,
            unmanaged_path=unmanaged_path,
            family_path=family_path,
            served_apis_path=served_apis_path,
            controller_path=controller_path,
        )

    def run_preflight(self, *args: str) -> subprocess.CompletedProcess:
        release = [] if {"-r", "--release"} & set(args) else ["-r", "miles-workbench-alice"]
        return run_workbench("install", "--dry-run", *release, *args, capture_output=True)

    def permission_queries(self, fake_cluster: dict) -> str:
        calls = fake_cluster["calls_path"].read_text().splitlines()
        return "\n".join(line for line in calls if line.startswith("auth"))

    def test_it_asks_for_exactly_the_rights_the_install_and_the_workflow_need(self, fake_cluster):
        """Pinned in full: a missing check is a false pass, and a spurious one turns away a legitimate installer."""
        expected = can_i_queries(
            merged_rules(
                cli.WORKFLOW_RULES,
                cli.CHART_RULES,
                cli.CHART_SERVICE_ACCOUNT_RULES,
                cli.CHART_RBAC_RULES,
                _FAKE_GRANTED_RULES,
                _FAKE_GRANTED_LWS_RULES,
            )
        )

        result = self.run_preflight("-n", "rl")
        calls = fake_cluster["calls_path"].read_text().splitlines()
        queries = {
            line.removeprefix("auth can-i ").removesuffix(" -n rl") for line in calls if line.startswith("auth")
        }

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Preflight checks passed" in result.stderr
        assert queries == expected
        assert "get namespace -- rl" in calls
        assert f"api-resources --api-group {cli.LWS_API_GROUP} -o name" in calls

    @requires_helm
    @pytest.mark.parametrize("release", ["miles-workbench-alice", "wb", "a" * 53])
    def test_the_name_it_computes_is_the_one_the_chart_renders(self, release):
        """The cli is the only thing that names this chart's objects, and the chart must take it verbatim."""
        computed = object_name(release)
        result = subprocess.run(
            ["helm", "template", release, str(CHART_DIR), "-n", NAMESPACE, "--set", f"objectName={computed}"],
            capture_output=True,
            text=True,
        )
        objects = [document for document in yaml.safe_load_all(result.stdout) if document is not None]
        rendered = named_object(objects, "Role", computed)["metadata"]["name"]

        assert computed == rendered

    @requires_helm
    def test_the_grants_it_checks_are_read_back_out_of_the_role_the_chart_ships(self):
        """Nothing restates the chart's rules, so the parse is the only thing that can lose one."""
        role = named_object(render(), "Role", RELEASE_NAME)
        granted = rbac_plan_of(yaml.safe_dump(role)).granted_rules

        assert granted
        assert cli.LWS_RESOURCE in granted
        assert all("/" not in resource or resource.count("/") == 1 for resource in granted)
        assert {verb for verbs in granted.values() for verb in verbs} >= {"get", "list", "watch"}

    def test_an_empty_namespace_passes_and_the_check_spans_the_whole_chart_family(self, fake_cluster):
        """Reinstalling over your own workbench is the normal case and must not read as a shared namespace."""
        result = self.run_preflight("-n", "rl")
        calls = fake_cluster["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "get all -n rl -l app.kubernetes.io/managed-by!=Helm -o name" in calls
        assert (
            "get all -n rl -l app.kubernetes.io/managed-by=Helm,"
            "app.kubernetes.io/name notin (miles-workbench,miles-run) -o name" in calls
        )

    def test_another_teams_helm_release_only_warns(self, fake_cluster):
        """A shared namespace is a legitimate setup, so naming the neighbour must not block the install."""
        (fake_cluster["foreign_path"]).write_text("statefulset.apps/someone-elses-database\n")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARN" in result.stderr
        assert "someone-elses-database" in result.stderr

    def test_a_resource_nobody_manages_only_warns(self, fake_cluster):
        """A hand-applied workload is worth naming, because this workbench would be able to delete it."""
        (fake_cluster["unmanaged_path"]).write_text("deployment.apps/hand-rolled\n")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARN" in result.stderr
        assert "hand-rolled" in result.stderr

    def test_a_live_miles_run_beside_the_workbench_passes(self, fake_cluster):
        """Reinstalling the workbench must not require tearing down the experiment it was installed to drive."""
        fake_cluster["family_path"].write_text(
            "statefulset.apps/myrun-miles-run-orchestrator\nservice/myrun-miles-run-engine\n"
        )

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "holds nothing but Miles releases" in result.stderr
        assert "myrun-miles-run-orchestrator" not in result.stderr

    def test_a_foreign_release_is_still_named_while_a_miles_run_is_live(self, fake_cluster):
        """The exemption is for the two Miles charts by name, not a blanket silence for a busy namespace."""
        fake_cluster["family_path"].write_text("statefulset.apps/myrun-miles-run-orchestrator\n")
        fake_cluster["foreign_path"].write_text("statefulset.apps/someone-elses-database\n")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "someone-elses-database" in result.stderr
        assert "myrun-miles-run-orchestrator" not in result.stderr

    def test_a_denied_rule_fails_the_run(self, fake_cluster):
        """Preflight exists to catch missing rights before install, so a denial must be a hard failure."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "configmaps" in result.stderr

    def test_escalate_and_bind_together_downgrade_a_missing_grant_to_a_warning(self, fake_cluster):
        """Those are the two verbs Kubernetes checks: escalate for the Role, bind for its RoleBinding."""
        (fake_cluster["denied_path"]).write_text("configmaps")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARN" in result.stderr

    @pytest.mark.parametrize("verb", ["bind", "escalate"])
    def test_a_right_restricted_to_this_release_counts(self, fake_cluster, verb):
        """helm applies the Role server-side, so Kubernetes checks both verbs against the object's name."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {verb}:roles.rbac.authorization.k8s.io")

        result = self.run_preflight("-n", "rl", "-r", "miles-workbench-alice")

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

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "may grant the workbench its Role" in result.stderr

    def test_a_grant_denied_without_escalate_is_a_failure(self, fake_cluster):
        """Without escalate the Role can only carry rules the installer holds, so the install would be rejected."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "may grant the workbench its Role" in result.stderr

    def test_missing_leaderworkerset_rights_name_the_admin_prerequisite(self, fake_cluster):
        """LWS rights come from a cluster admin, so that denial must point there rather than at the user."""
        (fake_cluster["denied_path"]).write_text(f"{LWS_RESOURCE} {NO_DELEGATION}")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "cluster admin" in result.stderr

    def test_an_unrelated_denial_does_not_blame_the_lws_prerequisite(self, fake_cluster):
        """Sending a user to their cluster admin over their own missing configmap rights wastes everyone's time."""
        (fake_cluster["denied_path"]).write_text(f"configmaps {NO_DELEGATION}")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "cluster admin" not in result.stderr

    def test_an_rbac_free_install_checks_only_what_it_installs(self, fake_cluster):
        """With rbac.create=false an admin pre-creates the identity, so none of that is the installer's to do."""
        (fake_cluster["denied_path"]).write_text("roles.rbac.authorization.k8s.io serviceaccounts configmaps")

        result = self.run_preflight("-n", "rl", "--no-rbac")
        permission_queries = self.permission_queries(fake_cluster)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "roles.rbac.authorization.k8s.io" not in permission_queries
        assert "serviceaccounts" not in permission_queries
        assert "configmaps" not in permission_queries

    def test_turning_off_leaderworkersets_drops_those_checks(self, fake_cluster):
        """A cluster without LWS installed cannot grant those rights, and must still get a workbench."""
        (fake_cluster["denied_path"]).write_text(LWS_RESOURCE)

        result = self.run_preflight("-n", "rl", "--no-lws")
        permission_queries = self.permission_queries(fake_cluster)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "leaderworkersets" not in permission_queries

    def test_a_values_file_that_switches_rbac_off_drops_those_checks_too(self, fake_cluster, tmp_path):
        """-f decides the install as surely as --no-rbac does, and checking for rights it will not need fails it."""
        values = tmp_path / "cluster.yaml"
        values.write_text("rbac:\n  create: false\n")
        (fake_cluster["denied_path"]).write_text("roles.rbac.authorization.k8s.io serviceaccounts")

        result = self.run_preflight("-n", "rl", "-f", str(values), "--set", "rbac.create=false")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "serviceaccounts" not in self.permission_queries(fake_cluster)

    def test_a_set_that_switches_rbac_back_on_restores_those_checks(self, fake_cluster):
        """--no-rbac then --set rbac.create=true is what helm installs, so it is what has to be checked."""
        (fake_cluster["denied_path"]).write_text(f"serviceaccounts {NO_DELEGATION}")

        result = self.run_preflight("-n", "rl", "--no-rbac", "--set", "rbac.create=true")

        assert result.returncode == 1
        assert "serviceaccounts" in result.stderr

    def test_a_set_that_switches_leaderworkersets_back_on_restores_those_checks(self, fake_cluster):
        """The Role helm renders would carry LWS rights, so skipping the LWS checks would install a broken one."""
        fake_cluster["served_apis_path"].write_text("")

        result = self.run_preflight("-n", "rl", "--no-lws", "--set", "rbac.leaderWorkerSets=true")

        assert result.returncode == 1
        assert f"the cluster serves {LWS_RESOURCE}" in result.stderr

    def test_values_that_do_not_render_stop_the_checks_from_guessing(self, fake_cluster, tmp_path):
        """A values file helm rejects fails the install, and reporting on defaults it will never use is a lie."""
        fake_cluster["bin_dir"].joinpath("helm").write_text(
            '#!/usr/bin/env bash\nif [ "$1" = "template" ]; then echo "Error: bad value" >&2; exit 1; fi\nexit 0\n'
        )
        fake_cluster["bin_dir"].joinpath("helm").chmod(0o755)

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "your values render the chart" in result.stderr

    @requires_helm
    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ([], (True, True)),
            (["--set", "rbac.create=false"], (False, False)),
            (["--set", "rbac.leaderWorkerSets=false"], (True, False)),
        ],
    )
    def test_the_plan_it_derives_is_what_the_chart_actually_renders(self, overrides, expected):
        """The whole point is to read helm's answer, so the parse has to hold against the real chart."""
        rendered = subprocess.run(
            ["helm", "template", "wb", str(CHART_DIR), "-n", NAMESPACE, *overrides],
            capture_output=True,
            text=True,
        )

        plan = rbac_plan_of(rendered.stdout)

        assert rendered.returncode == 0, rendered.stderr
        assert (plan.creates_role, plan.grants_leader_worker_sets) == expected

    def test_a_missing_namespace_fails(self, fake_cluster):
        """A mistyped -n is the likeliest user error, and every can-i answer would still look fine."""
        (fake_cluster["absent_path"]).write_text("rl")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "namespace rl exists" in result.stderr

    def test_an_lws_api_the_cluster_does_not_serve_fails(self, fake_cluster):
        """Discovery answers every authenticated caller, so an empty api group is a real, readable absence."""
        fake_cluster["served_apis_path"].write_text("")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert f"the cluster serves {LWS_RESOURCE}" in result.stderr

    def test_an_lws_controller_that_is_not_available_fails(self, fake_cluster):
        """The CRD alone accepts LeaderWorkerSets that nothing will ever turn into pods."""
        fake_cluster["controller_path"].write_text("False")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "lws-controller-manager is available" in result.stderr

    def test_an_lws_controller_nobody_may_read_is_reported_as_unverifiable(self, fake_cluster):
        """A namespace-scoped user cannot read lws-system, and a silent pass would certify a dead controller."""
        fake_cluster["forbidden_path"].write_text("lws-controller-manager")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "UNKNOWN" in result.stderr
        assert "lws-controller-manager is available" in result.stderr

    def test_a_lookup_that_is_merely_forbidden_is_not_reported_as_a_pass(self, fake_cluster):
        """A namespace-scoped user cannot read cluster-scoped objects, and unreadable is not the same as present."""
        (fake_cluster["forbidden_path"]).write_text("rl")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "UNKNOWN  namespace rl exists" in result.stderr
        assert "PASS  namespace rl exists" not in result.stderr

    def test_a_lookup_that_fails_for_any_other_reason_is_not_a_pass(self, fake_cluster):
        """Treating an unreadable answer as "present" is how a transient outage certifies a missing object."""
        (fake_cluster["broken_path"]).write_text("rl")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "namespace rl exists" in result.stderr

    def test_a_missing_object_whose_name_contains_forbidden_still_fails(self, fake_cluster):
        """The Forbidden test must key off the server's status, not off text that a name can contain."""
        (fake_cluster["absent_path"]).write_text("forbidden-ns")

        result = self.run_preflight("--namespace=forbidden-ns")

        assert result.returncode == 1
        assert "namespace forbidden-ns exists" in result.stderr

    def test_names_are_looked_up_after_the_option_terminator(self, fake_cluster):
        """kubectl would read a dash-leading name as flags, so lookups must pass it after "--"."""
        self.run_preflight("--namespace=-dashed")

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

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "cluster is reachable" in result.stderr
        assert "kubeconfig" in result.stderr
        assert "auth can-i" not in fake_cluster["calls_path"].read_text()

    def test_a_wholly_denied_run_points_at_the_namespace_and_context(self, fake_cluster):
        """A mistyped namespace denies everything, and a namespace-scoped user cannot look the namespace up."""
        (fake_cluster["deny_all_path"]).write_text("1")

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "confirm the namespace name and your kubectl context" in result.stderr

    def test_missing_binaries_are_reported_before_any_query(self, fake_cluster):
        """Without helm there is nothing to check, and the message must say which binary is missing."""
        (fake_cluster["bin_dir"] / "helm").unlink()

        result = self.run_preflight("-n", "rl")

        assert result.returncode == 1
        assert "helm is installed" in result.stderr
        assert fake_cluster["calls_path"].exists() is False

    def test_the_namespace_is_required(self, fake_cluster):
        """Every check is namespace-scoped, so a missing namespace is a usage error, not a failed check."""
        result = self.run_preflight()

        assert result.returncode == 2
        assert "--namespace" in result.stderr

    @pytest.mark.parametrize(
        "args", [["-n"], ["--namespace"], ["--namespace="], ["-n", "rl", "-r"], ["-n", "rl", "--release="]]
    )
    def test_a_flag_without_its_value_is_a_usage_error(self, fake_cluster, args):
        """A dangling flag must fail loudly instead of consuming the next flag or spinning in the parser."""
        result = self.run_preflight(*args)

        assert result.returncode == 2

    def test_an_unknown_argument_is_a_usage_error(self, fake_cluster):
        """A mistyped flag must not silently degrade into a partial check."""
        result = self.run_preflight("-n", "rl", "--deploy")

        assert result.returncode == 2
        assert "--deploy" in result.stderr

    def test_asking_for_help_succeeds(self, fake_cluster):
        """--help is not an error, and scripts wrapping the install read its exit code."""
        result = self.run_preflight("--help")

        assert result.returncode == 0
        assert "install" in result.stdout and "Usage" in result.stdout

    def test_captured_output_keeps_the_checks_in_order(self, fake_cluster):
        """Users redirect this into a ticket, so every line has to arrive in the order it was checked in."""
        result = run_workbench(
            "install", "--dry-run", "-n", "rl", "-r", "wb", stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        assert result.stdout.splitlines()[0] == "PASS  kubectl is installed"
