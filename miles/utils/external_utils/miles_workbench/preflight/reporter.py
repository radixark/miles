from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self) -> None:
        self.failed = False
        self.checks = 0
        self.failures = 0

    def report(self, ok: bool, message: str, counted: bool = True) -> None:
        if counted:
            self.checks += 1
        if ok:
            logger.info("PASS  %s", message)
            return

        logger.error("FAIL  %s", message)
        self.failed = True
        if counted:
            self.failures += 1

    def warn(self, message: str) -> None:
        logger.warning("WARN  %s", message)

    def report_unverifiable(self, message: str, reason: str) -> None:
        logger.info("UNKNOWN  %s: this account may not look, so nothing here confirms it (%s)", message, reason)

    @property
    def everything_was_denied(self) -> bool:
        return bool(self.checks) and self.failures == self.checks
