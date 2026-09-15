from __future__ import annotations

import subprocess

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    BinaryPresenceChecker,
    ClusterReachableChecker,
    LeaderWorkerSetApiChecker,
    LeaderWorkerSetControllerChecker,
    NamespaceListingChecker,
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    RoleDelegationChecker,
    Status,
    expand_resource_verbs,
)


class TestExpandResourceVerbs:
    def test_overlapping_rule_sets_keep_each_resource_verb_once_in_first_seen_order(self) -> None:
        """Overlapping rule sets keep each resource-verb pair once in first-seen order."""
        first_rules = {
            "pods": ("get", "list", "get"),
            "services": ("create",),
        }
        second_rules = {
            "pods": ("list", "watch"),
            "services": ("create", "delete"),
        }

        resource_verbs = expand_resource_verbs(first_rules, second_rules)

        assert resource_verbs == [
            ResourceVerb(verb="get", resource="pods"),
            ResourceVerb(verb="list", resource="pods"),
            ResourceVerb(verb="create", resource="services"),
            ResourceVerb(verb="watch", resource="pods"),
            ResourceVerb(verb="delete", resource="services"),
        ]


class TestBinaryPresenceChecker:
    @pytest.mark.parametrize(("found", "status"), [("/usr/bin/helm", Status.PASS), (None, Status.FAIL)])
    def test_binary_presence_reflects_path_lookup(
        self, monkeypatch: pytest.MonkeyPatch, found: str | None, status: Status
    ) -> None:
        """Binary presence passes exactly when the executable is on PATH."""
        monkeypatch.setattr(checkers_module.shutil, "which", lambda binary: found)

        result = BinaryPresenceChecker("helm").check()

        assert result == checkers_module.CheckResult(status=status, message="helm is installed")


class TestClusterReachableChecker:
    @pytest.mark.parametrize(
        ("returncode", "output", "status"),
        [
            (0, "developer", Status.PASS),
            (1, "authentication failed", Status.FAIL),
        ],
    )
    def test_identity_and_query_outcome_classify_cluster_access(
        self, monkeypatch: pytest.MonkeyPatch, returncode: int, output: str, status: Status
    ) -> None:
        """Reachability distinguishes a successful API response from a failed request."""
        _return_kubectl_result(monkeypatch, returncode=returncode, stdout=output)

        result = ClusterReachableChecker().check()

        assert result.status is status
        assert "cluster is reachable" in result.message


class TestResourceVerbAvailabilityChecker:
    def test_subresource_permission_uses_the_parent_resource_and_subresource_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A subresource permission is queried using kubectl's subresource protocol."""
        calls: list[tuple[str, ...]] = []

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="yes", stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))

        result = ResourceVerbAvailabilityChecker(
            namespace="rl", resource_verb=ResourceVerb(resource="pods/log", verb="get")
        ).check()

        assert result.status is Status.PASS
        assert calls == [("auth", "can-i", "get", "pods", "--subresource=log", "-n", "rl")]


class TestRoleDelegationChecker:
    def test_a_named_role_permission_is_accepted_after_the_resource_wide_probe_is_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Role delegation accepts a grant scoped to the exact role name."""
        answers = iter([1, 0])

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=list(args), returncode=next(answers), stdout="", stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))

        result = RoleDelegationChecker(namespace="rl", verb="bind", role="miles").check()

        assert result.status is Status.PASS


def _return_kubectl_result(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int, stdout: str = "", stderr: str = ""
) -> None:
    def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=list(args), returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))


class TestNamespaceListingChecker:
    def test_cluster_provided_namespace_objects_are_not_reported_as_foreign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kubernetes-created namespace objects are excluded while foreign objects remain."""
        _return_kubectl_result(
            monkeypatch,
            returncode=0,
            stdout=(
                "configmap/kube-root-ca.crt\n"
                "serviceaccount/default\n"
                "secret/default-token-7qk2m\n"
                "configmap/foreign-config\n"
                "secret/foreign-secret\n"
            ),
        )
        checker = NamespaceListingChecker(
            namespace="rl", kind="all", selector="managed-by!=Helm", release="miles-workbench"
        )

        result = checker.check()

        assert result.status is Status.PASS
        assert checker.foreign == ("configmap/foreign-config", "secret/foreign-secret")

    @pytest.mark.parametrize(
        ("error", "expected_message"),
        [
            ("the server doesn't have a resource type leaderworkerset", "does not serve leaderworkerset"),
            ("connection reset by peer", "listing leaderworkerset failed: connection reset by peer"),
        ],
    )
    def test_namespace_listing_failures_distinguish_unserved_resources_from_query_errors(
        self, monkeypatch: pytest.MonkeyPatch, error: str, expected_message: str
    ) -> None:
        """Unserved resources and ordinary listing failures retain distinct unknown explanations."""
        _return_kubectl_result(monkeypatch, returncode=1, stderr=error)
        checker = NamespaceListingChecker(
            namespace="rl", kind="leaderworkerset", selector="managed-by!=Helm", release="miles-workbench"
        )

        result = checker.check()

        assert result.status is Status.UNKNOWN
        assert expected_message in result.message


class TestLeaderWorkerSetCheckers:
    @pytest.mark.parametrize(
        ("checker", "error", "expected_status"),
        [
            (LeaderWorkerSetApiChecker(), "discovery unavailable", Status.UNKNOWN),
            (LeaderWorkerSetControllerChecker(), "connection reset by peer", Status.FAIL),
            (LeaderWorkerSetControllerChecker(), "deployments.apps is forbidden (Forbidden)", Status.UNKNOWN),
        ],
    )
    def test_lws_query_failures_preserve_unknown_and_fail_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        checker: LeaderWorkerSetApiChecker | LeaderWorkerSetControllerChecker,
        error: str,
        expected_status: Status,
    ) -> None:
        """LWS discovery and controller query failures preserve verifiability classifications."""
        _return_kubectl_result(monkeypatch, returncode=1, stderr=error)

        result = checker.check()

        assert result.status is expected_status
        assert error in result.message
