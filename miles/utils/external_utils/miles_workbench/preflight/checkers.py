from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

ROLES_RESOURCE = "roles.rbac.authorization.k8s.io"

_MAX_CONCURRENT_QUERIES = 32


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
