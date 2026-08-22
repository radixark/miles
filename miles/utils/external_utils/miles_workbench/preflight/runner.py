from __future__ import annotations

import logging

from miles.utils.external_utils.miles_workbench.naming import object_name
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    BaseChecker,
    BinaryPresenceChecker,
    CheckOutcome,
    CheckResult,
    ClusterReachableChecker,
    LeaderWorkerSetApiChecker,
    LeaderWorkerSetControllerChecker,
    NamespaceListingChecker,
    ResourcePresenceChecker,
    ResourceVerbAvailabilityChecker,
    RoleDelegationChecker,
    Status,
    expand_resource_verbs,
    parallel_execute_checkers,
)
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    CHART_FAMILY,
    CHART_RBAC_RULES,
    CHART_RULES,
    CHART_SERVICE_ACCOUNT_RULES,
    MANAGED_BY,
    NAMESPACE_KINDS,
    WORKFLOW_RULES,
)
from miles.utils.external_utils.miles_workbench.preflight.utils import Verdict, warn
from miles.utils.external_utils.miles_workbench.render import RbacPlan, rbac_plan_of, render_chart

logger = logging.getLogger(__name__)


def run_preflight_checks(args: InstallArgs) -> None:
    namespace = args.namespace
    verdict = Verdict()

    verdict.absorb(parallel_execute_checkers([BinaryPresenceChecker("kubectl"), BinaryPresenceChecker("helm")]))
    if verdict.failed:
        logger.error("Install the missing binaries before continuing")
        raise SystemExit(1)

    verdict.absorb(parallel_execute_checkers([ClusterReachableChecker()]))
    if verdict.failed:
        logger.error(
            "Fix your kubeconfig, credentials or network before reading anything below as a permission problem"
        )
        raise SystemExit(1)

    listings = _namespace_listing_checkers(namespace)
    verdict.absorb(
        parallel_execute_checkers(
            [
                ResourcePresenceChecker("namespace", namespace),
                *_verb_checkers(namespace, WORKFLOW_RULES),
                *listings,
            ]
        )
    )
    _warn_about_foreign_objects(namespace, listings)

    plan = _render_plan(args, verdict=verdict)

    install_rules = [CHART_RULES]
    if plan.creates_role:
        install_rules += [CHART_SERVICE_ACCOUNT_RULES, CHART_RBAC_RULES]
    install_checkers = _verb_checkers(namespace, *install_rules)
    grant_checkers = _verb_checkers(namespace, plan.granted_rules) if plan.creates_role else []
    cluster_checkers: list[BaseChecker] = []
    if plan.grants_leader_worker_sets:
        cluster_checkers += [LeaderWorkerSetApiChecker(), LeaderWorkerSetControllerChecker()]

    outcomes = parallel_execute_checkers([*cluster_checkers, *install_checkers, *grant_checkers])
    split = len(outcomes) - len(grant_checkers)
    verdict.absorb(outcomes[:split])
    grant_outcomes = outcomes[split:]
    verdict.observe(grant_outcomes)

    if plan.creates_role:
        _judge_the_grant(args, verdict=verdict, denied=_denied(grant_outcomes))

    verdict.announce()
    if verdict.failed:
        raise SystemExit(1)


def _judge_the_grant(args: InstallArgs, *, verdict: Verdict, denied: list[CheckOutcome]) -> None:
    message = f"may grant the workbench its Role in namespace {args.namespace}"
    if not denied:
        verdict.absorb_result(CheckResult(status=Status.PASS, message=message))
        return

    entries = _denied_entries(denied)
    delegation = parallel_execute_checkers(
        [RoleDelegationChecker(args.namespace, verb, object_name(args.release)) for verb in ("escalate", "bind")]
    )
    if all(outcome.result.status is Status.PASS for outcome in delegation):
        warn(
            f"you do not hold every rule the chart grants (denied: {', '.join(entries)}), but you have "
            "escalate and bind on roles, which is what Kubernetes checks when creating the Role and "
            "its RoleBinding"
        )
        return

    verdict.absorb_result(
        CheckResult(
            status=Status.FAIL,
            message=f"{message} (denied: {', '.join(entries)}; escalate and bind on roles would also do)",
        )
    )
    if any("leaderworkersets" in outcome.checker.resource for outcome in denied):
        logger.error(
            "A cluster admin must install the LWS CRDs and grant you the LeaderWorkerSet rights "
            "before this chart can grant them to the workbench"
        )


def _render_plan(args: InstallArgs, *, verdict: Verdict) -> RbacPlan:
    message = "your values render the chart"
    rendered = render_chart(args)
    if rendered.returncode != 0:
        verdict.absorb_result(
            CheckResult(status=Status.FAIL, message=f"{message} ({(rendered.stdout + rendered.stderr).strip()})")
        )
        return RbacPlan(creates_role=args.rbac)

    verdict.absorb_result(CheckResult(status=Status.PASS, message=message))
    return rbac_plan_of(rendered.stdout)


def _namespace_listing_checkers(namespace: str) -> list[NamespaceListingChecker]:
    family = ",".join(CHART_FAMILY)
    selectors = [
        f"app.kubernetes.io/managed-by!={MANAGED_BY}",
        f"app.kubernetes.io/managed-by={MANAGED_BY},app.kubernetes.io/name notin ({family})",
    ]
    return [NamespaceListingChecker(namespace, kind, selector) for kind in NAMESPACE_KINDS for selector in selectors]


def _warn_about_foreign_objects(namespace: str, listings: list[NamespaceListingChecker]) -> None:
    foreign: list[str] = []
    for listing in listings:
        foreign += [name for name in listing.foreign if name not in foreign]
    if not foreign:
        return

    warn(
        f"namespace {namespace} also holds {', '.join(foreign)}; the workbench Role covers the "
        f"whole namespace, so confirm this is the namespace you meant before sharing it with them"
    )


def _verb_checkers(namespace: str, *rule_sets: dict[str, tuple[str, ...]]) -> list[ResourceVerbAvailabilityChecker]:
    return [
        ResourceVerbAvailabilityChecker(namespace, resource_verb)
        for resource_verb in expand_resource_verbs(*rule_sets)
    ]


def _denied(outcomes: list[CheckOutcome]) -> list[CheckOutcome]:
    return [outcome for outcome in outcomes if outcome.result.status is Status.FAIL]


def _denied_entries(denied: list[CheckOutcome]) -> list[str]:
    by_resource: dict[str, list[str]] = {}
    for outcome in denied:
        by_resource.setdefault(outcome.checker.resource, []).append(outcome.checker.verb)
    return [f"{resource}({' '.join(verbs)})" for resource, verbs in by_resource.items()]
