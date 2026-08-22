from __future__ import annotations

import logging

from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    CheckOutcome,
    CheckResult,
    ResourceVerbAvailabilityChecker,
    Status,
)

logger = logging.getLogger(__name__)


class Verdict:
    def __init__(self) -> None:
        self.failed = False
        self._permission_checks = 0
        self._permission_failures = 0

    def absorb(self, outcomes: list[CheckOutcome]) -> None:
        for outcome in outcomes:
            self.absorb_result(outcome.result)
            if isinstance(outcome.checker, ResourceVerbAvailabilityChecker):
                self._permission_checks += 1
                if outcome.result.status is Status.FAIL:
                    self._permission_failures += 1

    def absorb_result(self, result: CheckResult) -> None:
        report(result)
        if result.status is Status.FAIL:
            self.failed = True

    def observe(self, outcomes: list[CheckOutcome]) -> None:
        for outcome in outcomes:
            report(outcome.result)

    def announce(self) -> None:
        if not self.failed:
            logger.info("Preflight checks passed")
            return

        if self._everything_was_denied:
            logger.error(
                "Every check was denied: confirm the namespace name and your kubectl context before "
                "treating this as missing RBAC"
            )
        logger.error("Preflight checks failed")

    @property
    def _everything_was_denied(self) -> bool:
        return bool(self._permission_checks) and self._permission_failures == self._permission_checks


def report(result: CheckResult) -> None:
    if result.status is Status.PASS:
        logger.info("PASS  %s", result.message)
        return
    if result.status is Status.UNKNOWN:
        logger.info("UNKNOWN  %s", result.message)
        return
    logger.error("FAIL  %s", result.message)


def warn(message: str) -> None:
    logger.warning("WARN  %s", message)
