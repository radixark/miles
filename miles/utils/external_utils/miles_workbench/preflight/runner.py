from __future__ import annotations

import logging

from miles.utils.external_utils.miles_workbench.naming import object_name
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.checker import Checker
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    CHART_RBAC_RULES,
    CHART_RULES,
    CHART_SERVICE_ACCOUNT_RULES,
    WORKFLOW_RULES,
)
from miles.utils.external_utils.miles_workbench.render import RbacPlan

logger = logging.getLogger(__name__)


def run_preflight_checks(args: InstallArgs) -> None:
    checker = Checker(namespace=args.namespace)

    checker.check_binary("kubectl")
    checker.check_binary("helm")
    if checker.failed:
        logger.error("Install the missing binaries before continuing")
        raise SystemExit(1)

    checker.check_cluster_reachable()
    checker.check_present("namespace", args.namespace)

    checker.check_rules("use the workbench", WORKFLOW_RULES)
    checker.check_namespace_holds_only()

    plan = checker.check_rbac_plan(args)
    if plan.grants_leader_worker_sets:
        checker.check_leader_worker_set_api()
        checker.check_leader_worker_set_controller()

    if plan.creates_role:
        _check_it_may_grant_the_role(checker, args, plan=plan)
    else:
        checker.check_rules("install the chart", CHART_RULES)

    _verdict(checker)


def _check_it_may_grant_the_role(checker: Checker, args: InstallArgs, *, plan: RbacPlan) -> None:
    checker.check_rules("install the chart", CHART_RULES, CHART_SERVICE_ACCOUNT_RULES, CHART_RBAC_RULES)

    message = f"may grant the workbench its Role in namespace {args.namespace}"

    if not (denied := checker.denied_rules(plan.granted_rules)):
        checker.report(True, message)
        return

    if checker.may_delegate_rules_it_does_not_hold(object_name(args.release)):
        checker.warn(
            f"you do not hold every rule the chart grants (denied: {', '.join(denied)}), but you have "
            "escalate and bind on roles, which is what Kubernetes checks when creating the Role and "
            "its RoleBinding"
        )
        return

    checker.report(False, f"{message} (denied: {', '.join(denied)}; escalate and bind on roles would also do)")
    if any("leaderworkersets" in entry for entry in denied):
        logger.error(
            "A cluster admin must install the LWS CRDs and grant you the LeaderWorkerSet rights "
            "before this chart can grant them to the workbench"
        )


def _verdict(checker: Checker) -> None:
    if not checker.failed:
        logger.info("Preflight checks passed")
        return

    if checker.everything_was_denied:
        logger.error(
            "Every check was denied: confirm the namespace name and your kubectl context before "
            "treating this as missing RBAC"
        )
    logger.error("Preflight checks failed")
    raise SystemExit(1)
