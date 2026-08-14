from __future__ import annotations

from miles.utils.external_utils.miles_workbench.naming import CHART_NAME

MANAGED_BY = "Helm"
CHART_FAMILY = (CHART_NAME, "miles-run")
LWS_API_GROUP = "leaderworkerset.x-k8s.io"
LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"
LWS_CONTROLLER_NAMESPACE = "lws-system"
LWS_CONTROLLER_DEPLOYMENT = "lws-controller-manager"
AVAILABLE_CONDITION = "jsonpath={.status.conditions[?(@.type=='Available')].status}"

NAMESPACE_KINDS = (
    "all",
    "configmap",
    "secret",
    "persistentvolumeclaim",
    "serviceaccount",
    "role.rbac.authorization.k8s.io",
    "rolebinding.rbac.authorization.k8s.io",
    "leaderworkerset.leaderworkerset.x-k8s.io",
)
UNSERVED_RESOURCE_MARKERS = ("doesn't have a resource type", "could not find the requested resource")
CLUSTER_PROVIDED_RESOURCES = ("configmap/kube-root-ca.crt", "serviceaccount/default")
DEFAULT_TOKEN_PREFIX = "secret/default-token-"

CHART_RULES: dict[str, tuple[str, ...]] = {
    "statefulsets.apps": ("create", "delete", "get", "patch"),
    "secrets": ("create", "delete", "get", "list", "update"),
}

CHART_SERVICE_ACCOUNT_RULES: dict[str, tuple[str, ...]] = {
    "serviceaccounts": ("create", "delete", "get", "patch"),
}

CHART_RBAC_RULES: dict[str, tuple[str, ...]] = {
    "roles.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
    "rolebindings.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
}

WORKFLOW_RULES: dict[str, tuple[str, ...]] = {
    "pods": ("get", "list"),
    "pods/exec": ("create",),
    "pods/log": ("get",),
    "statefulsets.apps": ("get", "list", "watch"),
}
