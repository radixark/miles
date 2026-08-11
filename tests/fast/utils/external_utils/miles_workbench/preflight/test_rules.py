from miles.utils.external_utils.miles_workbench.preflight.rules import (
    CHART_RBAC_RULES,
    CHART_RULES,
    CHART_SERVICE_ACCOUNT_RULES,
    WORKFLOW_RULES,
)


class TestBuiltinPermissionRules:
    def test_builtin_permission_rule_registry_is_complete(self) -> None:
        """The built-in registry contains exactly every externally required permission."""
        expected: dict[str, dict[str, tuple[str, ...]]] = {
            "chart": {
                "statefulsets.apps": ("create", "delete", "get", "patch"),
                "secrets": ("create", "delete", "get", "list", "update"),
            },
            "chart_service_account": {
                "serviceaccounts": ("create", "delete", "get", "patch"),
            },
            "chart_rbac": {
                "roles.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
                "rolebindings.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
            },
            "workflow": {
                "pods": ("get", "list"),
                "pods/exec": ("create",),
                "pods/log": ("get",),
                "statefulsets.apps": ("get", "list", "watch"),
            },
        }

        assert {
            "chart": CHART_RULES,
            "chart_service_account": CHART_SERVICE_ACCOUNT_RULES,
            "chart_rbac": CHART_RBAC_RULES,
            "workflow": WORKFLOW_RULES,
        } == expected
