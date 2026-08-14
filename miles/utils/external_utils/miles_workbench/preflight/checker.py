from __future__ import annotations

import logging
import shutil
import subprocess
from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.reporter import Reporter
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    AVAILABLE_CONDITION,
    CHART_FAMILY,
    CLUSTER_PROVIDED_RESOURCES,
    DEFAULT_TOKEN_PREFIX,
    LWS_API_GROUP,
    LWS_CONTROLLER_DEPLOYMENT,
    LWS_CONTROLLER_NAMESPACE,
    LWS_RESOURCE,
    MANAGED_BY,
    NAMESPACE_KINDS,
    UNSERVED_RESOURCE_MARKERS,
)
from miles.utils.external_utils.miles_workbench.render import RbacPlan, rbac_plan_of, render_chart

logger = logging.getLogger(__name__)

_ROLES_RESOURCE = "roles.rbac.authorization.k8s.io"


class _Answer(NamedTuple):
    ok: bool
    output: str


class Checker:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._reporter = Reporter()
        self._answered: dict[tuple[str, str], bool] = {}

    @property
    def failed(self) -> bool:
        return self._reporter.failed

    @property
    def everything_was_denied(self) -> bool:
        return self._reporter.everything_was_denied

    def report(self, ok: bool, message: str, counted: bool = True) -> None:
        self._reporter.report(ok, message, counted)

    def warn(self, message: str) -> None:
        self._reporter.warn(message)

    def check_binary(self, binary: str) -> None:
        self.report(shutil.which(binary) is not None, f"{binary} is installed", counted=False)

    def check_cluster_reachable(self) -> None:
        answer = self._query("get", "--raw", "/version")
        if answer.ok:
            return

        self.report(False, f"cluster is reachable and your credentials are accepted ({answer.output})", counted=False)
        logger.error(
            "Fix your kubeconfig, credentials or network before reading anything below as a permission problem"
        )
        raise SystemExit(1)

    def check_present(self, kind: str, name: str, namespace: str | None = None) -> None:
        scope = ["-n", namespace] if namespace else []
        answer = self._query("get", kind, *scope, "--", name)
        where = f" in namespace {namespace}" if namespace else ""
        message = f"{kind} {name} exists{where}"
        if not answer.ok:
            self._report_failed_query(message, answer, unverifiable="(Forbidden)" in answer.output, counted=False)

    def check_namespace_holds_only(self) -> None:
        message = f"namespace {self.namespace} holds nothing but Miles releases"
        family = ",".join(CHART_FAMILY)
        selectors = [
            f"app.kubernetes.io/managed-by!={MANAGED_BY}",
            f"app.kubernetes.io/managed-by={MANAGED_BY},app.kubernetes.io/name notin ({family})",
        ]

        foreign: list[str] = []
        for kind in NAMESPACE_KINDS:
            for selector in selectors:
                answer = self._query("get", kind, "-n", self.namespace, "-l", selector, "-o", "name")
                if not answer.ok:
                    if any(marker in answer.output for marker in UNSERVED_RESOURCE_MARKERS):
                        break
                    self.warn(f"could not check whether {message} (listing {kind} failed: {answer.output})")
                    return
                foreign += [
                    name for name in answer.output.split() if name not in foreign and not _is_cluster_provided(name)
                ]

        if foreign:
            self.warn(
                f"namespace {self.namespace} also holds {', '.join(foreign)}; the workbench Role covers the "
                f"whole namespace, so confirm this is the namespace you meant before sharing it with them"
            )
            return
        self.report(True, message, counted=False)

    def check_rbac_plan(self, args: InstallArgs) -> RbacPlan:
        message = "your values render the chart"
        rendered = render_chart(args)
        if rendered.returncode != 0:
            self.report(False, f"{message} ({(rendered.stdout + rendered.stderr).strip()})", counted=False)
            return RbacPlan(creates_role=args.rbac)

        self.report(True, message, counted=False)
        return rbac_plan_of(rendered.stdout)

    def check_leader_worker_set_api(self) -> None:
        message = f"the cluster serves {LWS_RESOURCE}"
        answer = self._query("api-resources", "--api-group", LWS_API_GROUP, "-o", "name")
        if not answer.ok:
            self._report_failed_query(message, answer, unverifiable=True)
            return
        if LWS_RESOURCE not in answer.output.split():
            self.report(
                False,
                f"{message} (api discovery served {answer.output or 'nothing'} in {LWS_API_GROUP})",
                counted=False,
            )
            return
        self.report(True, message, counted=False)

    def check_leader_worker_set_controller(self) -> None:
        message = f"deployment {LWS_CONTROLLER_DEPLOYMENT} is available in namespace {LWS_CONTROLLER_NAMESPACE}"
        answer = self._query(
            "get",
            "deployment.apps",
            "-n",
            LWS_CONTROLLER_NAMESPACE,
            "-o",
            AVAILABLE_CONDITION,
            "--",
            LWS_CONTROLLER_DEPLOYMENT,
        )
        if not answer.ok:
            self._report_failed_query(message, answer, unverifiable="(Forbidden)" in answer.output, counted=False)
            return
        if answer.output != "True":
            self.report(
                False, f"{message} (the Available condition reads {answer.output or 'nothing'})", counted=False
            )
            return
        self.report(True, message, counted=False)

    def check_rules(self, what: str, *rule_sets: dict[str, tuple[str, ...]]) -> None:
        denied = self.denied_rules(*rule_sets)
        message = f"may {what} in namespace {self.namespace}"
        if denied:
            self.report(False, f"{message} (denied: {', '.join(denied)})")
            return
        self.report(True, message)

    def denied_rules(self, *rule_sets: dict[str, tuple[str, ...]]) -> list[str]:
        denied = []
        for rules in rule_sets:
            for resource, verbs in rules.items():
                if missing := [verb for verb in verbs if not self.can_i(verb, resource)]:
                    denied.append(f"{resource}({' '.join(missing)})")
        return denied

    def may_delegate_rules_it_does_not_hold(self, role: str) -> bool:
        return all(self._holds_on_roles(verb, role) for verb in ("escalate", "bind"))

    def can_i(self, verb: str, resource: str) -> bool:
        if (answered := self._answered.get((verb, resource))) is not None:
            return answered

        target, _, subresource = resource.partition("/")
        args = ["auth", "can-i", verb, target]
        if subresource:
            args.append(f"--subresource={subresource}")
        args += ["-n", self.namespace]
        self._answered[(verb, resource)] = self._query(*args).ok
        return self._answered[(verb, resource)]

    def kubectl(self, *args: str) -> subprocess.CompletedProcess[str]:
        return Kubectl.run_raw(*args)

    def _holds_on_roles(self, verb: str, role: str) -> bool:
        if self.can_i(verb, _ROLES_RESOURCE):
            return True
        return self._query("auth", "can-i", verb, f"{_ROLES_RESOURCE}/{role}", "-n", self.namespace).ok

    def _query(self, *args: str) -> _Answer:
        result = self.kubectl(*args)
        return _Answer(ok=result.returncode == 0, output=(result.stdout + result.stderr).strip())

    def _report_failed_query(self, message: str, answer: _Answer, *, unverifiable: bool, counted: bool = True) -> None:
        if unverifiable:
            self._reporter.report_unverifiable(message, answer.output)
            return
        self.report(False, f"{message} ({answer.output})", counted=counted)


def _is_cluster_provided(name: str) -> bool:
    return name in CLUSTER_PROVIDED_RESOURCES or name.startswith(DEFAULT_TOKEN_PREFIX)
