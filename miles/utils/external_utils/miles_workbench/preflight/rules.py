from __future__ import annotations

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
