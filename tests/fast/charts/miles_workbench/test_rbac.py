from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    objects_of_kind,
    pod_spec,
    render,
    requires_helm,
    single_object_of_kind,
)


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
        rules = single_object_of_kind(render(), "Role")["rules"]
        granted = {
            (group, resource): set(rule["verbs"])
            for rule in rules
            for group in rule["apiGroups"]
            for resource in rule["resources"]
        }
        write = {"create", "delete", "get", "list", "patch", "update", "watch"}

        assert granted == {
            ("", "configmaps"): write,
            ("", "secrets"): write,
            ("", "serviceaccounts"): write,
            ("", "services"): write,
            ("", "pods"): {"delete", "get", "list", "watch"},
            ("", "pods/exec"): {"create"},
            ("", "pods/log"): {"get"},
            ("", "events"): {"get", "list", "watch"},
            ("", "persistentvolumeclaims"): {"get", "list", "watch"},
            ("apps", "statefulsets"): write,
            ("batch", "jobs"): write,
            ("leaderworkerset.x-k8s.io", "leaderworkersets"): write,
        }

    def test_the_role_can_neither_grant_nor_escalate_permissions(self):
        """A workbench that can write RBAC or reach cluster scope would be as dangerous as binding admin."""
        rules = single_object_of_kind(render(), "Role")["rules"]
        groups = {group for rule in rules for group in rule["apiGroups"]}
        resources = {resource for rule in rules for resource in rule["resources"]}

        assert "rbac.authorization.k8s.io" not in groups
        assert not {"roles", "rolebindings", "clusterroles", "clusterrolebindings"} & resources
        assert "*" not in resources and "*" not in groups
        assert not any("*" in rule["verbs"] for rule in rules)

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
