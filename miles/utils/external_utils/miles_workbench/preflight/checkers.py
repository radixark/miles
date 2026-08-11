from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    AVAILABLE_CONDITION,
    CLUSTER_PROVIDED_RESOURCES,
    DEFAULT_TOKEN_PREFIX,
    LWS_API_GROUP,
    LWS_CONTROLLER_DEPLOYMENT,
    LWS_CONTROLLER_NAMESPACE,
    LWS_RESOURCE,
    UNSERVED_RESOURCE_MARKERS,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

ROLES_RESOURCE = "roles.rbac.authorization.k8s.io"

_MAX_CONCURRENT_QUERIES = 32
_FORBIDDEN_MARKER = "(Forbidden)"


# ============================== vocabulary ==============================


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class CheckResult(FrozenStrictBaseModel):
    status: Status
    message: str


class ResourceVerb(FrozenStrictBaseModel):
    verb: str
    resource: str


class BaseChecker:
    def check(self) -> CheckResult:
        raise NotImplementedError


class CheckOutcome(NamedTuple):
    checker: BaseChecker
    result: CheckResult


def parallel_execute_checkers(checkers: Sequence[BaseChecker]) -> list[CheckOutcome]:
    if not checkers:
        return []

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_QUERIES) as pool:
        results = pool.map(lambda checker: checker.check(), checkers)
        return [CheckOutcome(checker, result) for checker, result in zip(checkers, results, strict=True)]


def expand_resource_verbs(*rule_sets: dict[str, tuple[str, ...]]) -> list[ResourceVerb]:
    seen: dict[ResourceVerb, None] = {}
    for rules in rule_sets:
        for resource, verbs in rules.items():
            for verb in verbs:
                seen.setdefault(ResourceVerb(verb=verb, resource=resource))
    return list(seen)


# =============================== checkers ===============================


class BinaryPresenceChecker(BaseChecker):
    def __init__(self, binary: str) -> None:
        self.binary = binary

    def check(self) -> CheckResult:
        message = f"{self.binary} is installed"
        found = shutil.which(self.binary) is not None
        return CheckResult(status=Status.PASS if found else Status.FAIL, message=message)


class ClusterReachableChecker(BaseChecker):
    def check(self) -> CheckResult:
        message = "cluster is reachable and your credentials are accepted"
        answer = _query("get", "--raw", "/version")
        if not answer.ok:
            return CheckResult(status=Status.FAIL, message=f"{message} ({answer.output})")
        return CheckResult(status=Status.PASS, message=message)


class ResourcePresenceChecker(BaseChecker):
    def __init__(self, kind: str, name: str, namespace: str | None = None) -> None:
        self.kind = kind
        self.name = name
        self.namespace = namespace

    def check(self) -> CheckResult:
        scope = ["-n", self.namespace] if self.namespace else []
        answer = _query("get", self.kind, *scope, "--", self.name)
        where = f" in namespace {self.namespace}" if self.namespace else ""
        message = f"{self.kind} {self.name} exists{where}"
        if answer.ok:
            return CheckResult(status=Status.PASS, message=message)
        return _failed_query_result(message, answer, unverifiable=_FORBIDDEN_MARKER in answer.output)


class NamespaceListingChecker(BaseChecker):
    def __init__(self, namespace: str, kind: str, selector: str) -> None:
        self.namespace = namespace
        self.kind = kind
        self.selector = selector
        self.foreign: tuple[str, ...] = ()

    def check(self) -> CheckResult:
        message = f"namespace {self.namespace} holds nothing but Miles releases ({self.kind} matching {self.selector})"
        answer = _query("get", self.kind, "-n", self.namespace, "-l", self.selector, "-o", "name")
        if not answer.ok:
            if any(marker in answer.output for marker in UNSERVED_RESOURCE_MARKERS):
                return CheckResult(
                    status=Status.UNKNOWN, message=f"the cluster does not serve {self.kind}, so nothing lists it"
                )
            return CheckResult(
                status=Status.UNKNOWN,
                message=f"could not check whether namespace {self.namespace} holds nothing but Miles releases "
                f"(listing {self.kind} failed: {answer.output})",
            )

        self.foreign = tuple(name for name in answer.output.split() if not _is_cluster_provided(name))
        return CheckResult(status=Status.PASS, message=message)


class ResourceVerbAvailabilityChecker(BaseChecker):
    def __init__(self, namespace: str, resource_verb: ResourceVerb) -> None:
        self.namespace = namespace
        self.resource_verb = resource_verb

    @property
    def resource(self) -> str:
        return self.resource_verb.resource

    @property
    def verb(self) -> str:
        return self.resource_verb.verb

    def check(self) -> CheckResult:
        target, _, subresource = self.resource.partition("/")
        args = ["auth", "can-i", self.verb, target]
        if subresource:
            args.append(f"--subresource={subresource}")
        args += ["-n", self.namespace]

        message = f"may {self.verb} {self.resource} in namespace {self.namespace}"
        return CheckResult(status=Status.PASS if _query(*args).ok else Status.FAIL, message=message)


class RoleDelegationChecker(BaseChecker):
    def __init__(self, namespace: str, verb: str, role: str) -> None:
        self.namespace = namespace
        self.verb = verb
        self.role = role

    def check(self) -> CheckResult:
        message = f"may {self.verb} roles in namespace {self.namespace}"
        for target in (ROLES_RESOURCE, f"{ROLES_RESOURCE}/{self.role}"):
            if _query("auth", "can-i", self.verb, target, "-n", self.namespace).ok:
                return CheckResult(status=Status.PASS, message=message)
        return CheckResult(status=Status.FAIL, message=message)


class LeaderWorkerSetApiChecker(BaseChecker):
    def check(self) -> CheckResult:
        message = f"the cluster serves {LWS_RESOURCE}"
        answer = _query("api-resources", "--api-group", LWS_API_GROUP, "-o", "name")
        if not answer.ok:
            return _failed_query_result(message, answer, unverifiable=True)
        if LWS_RESOURCE not in answer.output.split():
            served = answer.output or "nothing"
            return CheckResult(
                status=Status.FAIL, message=f"{message} (api discovery served {served} in {LWS_API_GROUP})"
            )
        return CheckResult(status=Status.PASS, message=message)


class LeaderWorkerSetControllerChecker(BaseChecker):
    def check(self) -> CheckResult:
        message = f"deployment {LWS_CONTROLLER_DEPLOYMENT} is available in namespace {LWS_CONTROLLER_NAMESPACE}"
        answer = _query(
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
            return _failed_query_result(message, answer, unverifiable=_FORBIDDEN_MARKER in answer.output)
        if answer.output != "True":
            condition = answer.output or "nothing"
            return CheckResult(status=Status.FAIL, message=f"{message} (the Available condition reads {condition})")
        return CheckResult(status=Status.PASS, message=message)


# ================================ queries ===============================


class _Answer(FrozenStrictBaseModel):
    ok: bool
    output: str


def _query(*args: str) -> _Answer:
    result = Kubectl.run_raw(*args)
    return _Answer(ok=result.returncode == 0, output=(result.stdout + result.stderr).strip())


def _failed_query_result(message: str, answer: _Answer, *, unverifiable: bool) -> CheckResult:
    if unverifiable:
        return CheckResult(
            status=Status.UNKNOWN,
            message=f"{message}: this account may not look, so nothing here confirms it ({answer.output})",
        )
    return CheckResult(status=Status.FAIL, message=f"{message} ({answer.output})")


def _is_cluster_provided(name: str) -> bool:
    return name in CLUSTER_PROVIDED_RESOURCES or name.startswith(DEFAULT_TOKEN_PREFIX)
