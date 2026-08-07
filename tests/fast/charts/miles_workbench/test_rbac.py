import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    objects_of_kind,
    pod_spec,
    render,
    render_run,
    requires_helm,
    single_object_of_kind,
)

COLOCATED_RUN = (
    "--set-json",
    f'run.inferenceEngines={json.dumps([{"name": "engine", "replicas": 1, "size": 1, "command": ["python"]}])}',
    "--set-json",
    f'run.trainers={json.dumps([{"name": "trainer", "replicas": 1, "size": 1, "command": ["python"]}])}',
    "--set",
    "run.colocate.enabled=true",
    "--set",
    "run.colocate.enginePool=engine",
    "--set",
    "run.colocate.trainerPool=trainer",
)


def granted_verbs(role: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    return {
        (group, resource): set(rule["verbs"])
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
    }


@requires_helm
class TestRbacTemplates:
    def test_the_workbench_gets_its_own_namespaced_role(self):
        """The account is granted a Role this chart ships, never a cluster-wide role such as the built-in admin."""
        objects = render()
        role = single_object_of_kind(objects, "Role")
        binding = single_object_of_kind(objects, "RoleBinding")

        assert single_object_of_kind(objects, "ServiceAccount")["metadata"]["name"] == RELEASE_NAME
        assert objects_of_kind(objects, "ClusterRole") == []
        assert objects_of_kind(objects, "ClusterRoleBinding") == []
        assert binding["roleRef"] == dict(apiGroup="rbac.authorization.k8s.io", kind="Role", name=RELEASE_NAME)
        assert binding["subjects"] == [dict(kind="ServiceAccount", name=RELEASE_NAME, namespace=NAMESPACE)]
        assert role["metadata"]["name"] == RELEASE_NAME

    def test_the_role_stays_inside_what_installing_miles_run_needs(self):
        """Least privilege is the point of shipping our own Role, so the rule set is pinned in full."""
        write = {"create", "delete", "get", "list", "patch", "update", "watch"}

        assert granted_verbs(single_object_of_kind(render(), "Role")) == {
            ("", "configmaps"): write,
            ("", "secrets"): write,
            ("", "serviceaccounts"): write,
            ("", "services"): write,
            ("", "pods"): {"delete", "get", "list", "patch", "update", "watch"},
            ("", "pods/exec"): {"create"},
            ("", "pods/log"): {"get"},
            ("", "events"): {"get", "list", "watch"},
            ("", "persistentvolumeclaims"): {"get", "list", "watch"},
            ("apps", "deployments"): write,
            ("apps", "statefulsets"): write,
            ("batch", "jobs"): write,
            ("rbac.authorization.k8s.io", "roles"): write,
            ("rbac.authorization.k8s.io", "rolebindings"): write,
            ("leaderworkerset.x-k8s.io", "leaderworkersets"): write,
        }

    def test_the_role_covers_every_object_kind_miles_run_installs(self):
        """A kind miles-run renders but the Role omits turns every colocated install into an apiserver rejection."""
        granted = granted_verbs(single_object_of_kind(render(), "Role"))
        installed = {
            ("" if group in ("", "v1") else group, obj["kind"].lower() + "s")
            for obj in render_run(*COLOCATED_RUN)
            for group in [obj["apiVersion"].rpartition("/")[0]]
        }

        assert installed <= set(granted), sorted(installed - set(granted))
        assert all("create" in granted[key] for key in installed)

    def test_the_role_is_a_superset_of_the_role_miles_run_asks_it_to_create(self):
        """Kubernetes refuses a Role or RoleBinding carrying rules its creator does not already hold."""
        granted = granted_verbs(single_object_of_kind(render(), "Role"))
        created = [granted_verbs(role) for role in objects_of_kind(render_run(*COLOCATED_RUN), "Role")]

        assert created
        for rules in created:
            assert all(verbs <= granted.get(key, set()) for key, verbs in rules.items())

    def test_the_role_can_neither_escalate_nor_reach_cluster_scope(self):
        """It may write namespaced RBAC only because it holds those rules; escalate or bind would lift that ceiling."""
        rules = single_object_of_kind(render(), "Role")["rules"]
        resources = {resource for rule in rules for resource in rule["resources"]}
        verbs = {verb for rule in rules for verb in rule["verbs"]}

        assert not {"clusterroles", "clusterrolebindings"} & resources
        assert not {"escalate", "bind", "impersonate", "*"} & verbs
        assert "*" not in resources
        assert "*" not in {group for rule in rules for group in rule["apiGroups"]}

    def test_the_leaderworkerset_rules_can_be_turned_off(self):
        """A cluster without LWS installed cannot grant rights over it, and must still get a workbench."""
        rules = single_object_of_kind(render("--set", "rbac.leaderWorkerSets=false"), "Role")["rules"]
        groups = {group for rule in rules for group in rule["apiGroups"]}

        assert "leaderworkerset.x-k8s.io" not in groups

    def test_rbac_create_false_only_references_a_preexisting_account(self):
        """Strict clusters pre-create the identity; the chart then creates no RBAC object at all."""
        objects = render("--set", "rbac.create=false", "--set", "serviceAccount.name=preexisting")

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []
        assert pod_spec(objects)["serviceAccountName"] == "preexisting"

    def test_an_overridden_service_account_name_is_used_by_every_object(self):
        """A renamed account must stay consistent across the pod and its bindings, or the pod silently loses rights."""
        objects = render("--set", "serviceAccount.name=custom-sa")
        binding = single_object_of_kind(objects, "RoleBinding")

        assert single_object_of_kind(objects, "ServiceAccount")["metadata"]["name"] == "custom-sa"
        assert binding["subjects"][0]["name"] == "custom-sa"
        assert pod_spec(objects)["serviceAccountName"] == "custom-sa"
