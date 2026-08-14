import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    UNINSTALLER_SERVICE_ACCOUNT,
    named_object,
    objects_of_kind,
    pod_spec,
    render,
    render_run,
    requires_helm,
)

_POOL_ENTRIES = {
    name: [{"name": name, "objectName": f"myrun-miles-run-{name}", "replicas": 1, "size": 1, "command": ["python"]}]
    for name in ("engine", "trainer")
}

UNINSTALLABLE_RUN = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(_POOL_ENTRIES['engine'])}",
    "--set-json",
    f"run.trainerEngines={json.dumps(_POOL_ENTRIES['trainer'])}",
    "--set",
    "run.autoUninstall.enabled=true",
)

_PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": "myrun",
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "gpu_offset": 0,
            },
        }
    ],
}

COLOCATED_RUN = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(_POOL_ENTRIES['engine'])}",
    "--set-json",
    f"run.trainerEngines={json.dumps(_POOL_ENTRIES['trainer'])}",
    "--set-json",
    f"run.colocate={json.dumps(_PAIRING_CONFIG)}",
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
        role = named_object(objects, "Role", RELEASE_NAME)
        binding = named_object(objects, "RoleBinding", RELEASE_NAME)

        assert named_object(objects, "ServiceAccount", RELEASE_NAME)["metadata"]["name"] == RELEASE_NAME
        assert objects_of_kind(objects, "ClusterRole") == []
        assert objects_of_kind(objects, "ClusterRoleBinding") == []
        assert binding["roleRef"] == dict(apiGroup="rbac.authorization.k8s.io", kind="Role", name=RELEASE_NAME)
        assert binding["subjects"] == [dict(kind="ServiceAccount", name=RELEASE_NAME, namespace=NAMESPACE)]
        assert role["metadata"]["name"] == RELEASE_NAME

    def test_the_role_stays_inside_what_installing_miles_run_needs(self):
        """Least privilege is the point of shipping our own Role, so the rule set is pinned in full."""
        write = {"create", "delete", "get", "list", "patch", "update", "watch"}

        assert granted_verbs(named_object(render(), "Role", RELEASE_NAME)) == {
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
        granted = granted_verbs(named_object(render(), "Role", RELEASE_NAME))
        installed = {
            ("" if group in ("", "v1") else group, obj["kind"].lower() + "s")
            for obj in render_run(*COLOCATED_RUN)
            for group in [obj["apiVersion"].rpartition("/")[0]]
        }

        assert installed <= set(granted), sorted(installed - set(granted))
        assert all("create" in granted[key] for key in installed)

    def test_the_uninstaller_account_can_delete_a_run_and_nothing_else(self):
        """A run's escape job runs as this account, and every verb beyond deletion is one it does not need."""
        role = named_object(render(), "Role", UNINSTALLER_SERVICE_ACCOUNT)
        granted = granted_verbs(role)

        assert {verb for verbs in granted.values() for verb in verbs} == {"get", "list", "delete"}
        assert set(granted) == {
            ("", "configmaps"),
            ("", "secrets"),
            ("", "serviceaccounts"),
            ("", "services"),
            ("", "pods"),
            ("apps", "deployments"),
            ("apps", "statefulsets"),
            ("batch", "jobs"),
            ("rbac.authorization.k8s.io", "roles"),
            ("rbac.authorization.k8s.io", "rolebindings"),
            ("leaderworkerset.x-k8s.io", "leaderworkersets"),
        }

    def test_the_uninstaller_is_one_account_per_namespace_under_a_fixed_name(self):
        """A run finds it without knowing which workbench release created it, so the name may not be derived."""
        objects = render("--set", "objectName=another-workbench")

        assert (
            named_object(objects, "ServiceAccount", UNINSTALLER_SERVICE_ACCOUNT)["metadata"]["namespace"] == NAMESPACE
        )
        assert named_object(objects, "RoleBinding", UNINSTALLER_SERVICE_ACCOUNT)["roleRef"]["name"] == (
            UNINSTALLER_SERVICE_ACCOUNT
        )

    def test_the_uninstaller_covers_every_kind_a_run_release_owns(self):
        """helm uninstall stops at the first kind it may not delete, and leaves the release half removed."""
        granted = granted_verbs(named_object(render(), "Role", UNINSTALLER_SERVICE_ACCOUNT))
        installed = {
            ("" if group in ("", "v1") else group, obj["kind"].lower() + "s")
            for obj in render_run(*UNINSTALLABLE_RUN)
            for group in [obj["apiVersion"].rpartition("/")[0]]
        }

        assert installed <= set(granted), sorted(installed - set(granted))
        assert all("delete" in granted[key] for key in installed)

    def test_the_uninstaller_leaderworkerset_rules_follow_the_workbench_ones(self):
        """A cluster without the LWS CRDs cannot grant those rights to either account."""
        role = named_object(render("--set", "rbac.leaderWorkerSets=false"), "Role", UNINSTALLER_SERVICE_ACCOUNT)

        assert "leaderworkerset.x-k8s.io" not in {group for rule in role["rules"] for group in rule["apiGroups"]}

    def test_the_role_is_a_superset_of_the_role_miles_run_asks_it_to_create(self):
        """Kubernetes refuses a Role or RoleBinding carrying rules its creator does not already hold."""
        granted = granted_verbs(named_object(render(), "Role", RELEASE_NAME))
        created = [granted_verbs(role) for role in objects_of_kind(render_run(*COLOCATED_RUN), "Role")]

        assert created
        for rules in created:
            assert all(verbs <= granted.get(key, set()) for key, verbs in rules.items())

    def test_the_role_can_neither_escalate_nor_reach_cluster_scope(self):
        """It may write namespaced RBAC only because it holds those rules; escalate or bind would lift that ceiling."""
        rules = named_object(render(), "Role", RELEASE_NAME)["rules"]
        resources = {resource for rule in rules for resource in rule["resources"]}
        verbs = {verb for rule in rules for verb in rule["verbs"]}

        assert not {"clusterroles", "clusterrolebindings"} & resources
        assert not {"escalate", "bind", "impersonate", "*"} & verbs
        assert "*" not in resources
        assert "*" not in {group for rule in rules for group in rule["apiGroups"]}

    def test_the_leaderworkerset_rules_can_be_turned_off(self):
        """A cluster without LWS installed cannot grant rights over it, and must still get a workbench."""
        rules = named_object(render("--set", "rbac.leaderWorkerSets=false"), "Role", RELEASE_NAME)["rules"]
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
        binding = named_object(objects, "RoleBinding", RELEASE_NAME)

        assert named_object(objects, "ServiceAccount", "custom-sa")["metadata"]["name"] == "custom-sa"
        assert binding["subjects"][0]["name"] == "custom-sa"
        assert pod_spec(objects)["serviceAccountName"] == "custom-sa"
