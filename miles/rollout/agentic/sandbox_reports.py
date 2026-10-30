"""Agent code reports the health of its sandboxes to the run's observers.

Works with any sandbox backend: ``provider`` and ``sandbox_id`` are free-form
labels. Inside a ``subproc`` agent call a report goes to the process that made
the call, which hands it on with ``push_sandbox_reports``.
"""

from __future__ import annotations

import logging
import time

from miles.rollout.agentic.agent_function import report_to_caller
from miles.utils.logging_utils import RateLimitedWarner
from miles.utils.observers import SANDBOX_STATUSES, Observers, SandboxReport, SandboxStatus

logger = logging.getLogger(__name__)

_warner = RateLimitedWarner(logger)


def report_sandbox(
    sandbox_key: str,
    *,
    task: str | None = None,
    provider: str | None = None,
    sandbox_id: str | None = None,
    status: SandboxStatus = "healthy",
    error: str | None = None,
) -> None:
    """Report one sandbox's health to the run's observers.

    Call it when the sandbox starts, after each command it runs (the call time
    becomes its "last executed" time), on an error and on teardown
    (``status="finished"``). Works in the rollout manager process and in
    ``subproc`` agent calls; a no-op when the run has no observers."""
    try:
        if status not in SANDBOX_STATUSES:
            raise ValueError(f"{status=} must be one of {SANDBOX_STATUSES}")
        report = SandboxReport(
            ts=time.time(),
            sandbox_key=str(sandbox_key),
            status=status,
            task=_optional_str(task),
            provider=_optional_str(provider),
            sandbox_id=_optional_str(sandbox_id),
            error=_optional_str(error),
        )
        if not report_to_caller(report):
            push_sandbox_reports([report])
    except Exception as e:
        _warner.warn(f"sandbox report dropped: {e!r}")


def push_sandbox_reports(reports: list[SandboxReport]) -> None:
    """Hand sandbox reports to this process's observers, e.g. those a ``subproc`` agent call made."""
    observers = Observers()
    for report in reports:
        observers.sandbox_report(report)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
