import logging

import pytest

from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    CheckOutcome,
    CheckResult,
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    Status,
)
from miles.utils.external_utils.miles_workbench.preflight.utils import Verdict, report, warn


def _permission_outcome(status: Status) -> CheckOutcome:
    checker = ResourceVerbAvailabilityChecker(namespace="rl", resource_verb=ResourceVerb(resource="pods", verb="get"))
    return CheckOutcome(checker=checker, result=CheckResult(status=status, message="permission"))


class TestVerdict:
    def test_absorb_marks_failures_while_observe_only_reports_them(self, caplog: pytest.LogCaptureFixture) -> None:
        """Absorbed failures gate installation while observed failures remain informational."""
        result = CheckResult(status=Status.FAIL, message="failed check")
        absorbed = Verdict()
        observed = Verdict()

        with caplog.at_level(logging.INFO):
            absorbed.absorb_result(result)
            observed.observe(
                [
                    CheckOutcome(
                        checker=ResourceVerbAvailabilityChecker(
                            namespace="rl", resource_verb=ResourceVerb(resource="pods", verb="get")
                        ),
                        result=result,
                    )
                ]
            )

        assert absorbed.failed is True
        assert observed.failed is False
        assert caplog.text.count("FAIL  failed check") == 2

    def test_announce_explains_when_every_permission_probe_was_denied(self, caplog: pytest.LogCaptureFixture) -> None:
        """A uniformly denied permission set emits the wrong-context diagnostic."""
        verdict = Verdict()
        verdict.absorb([_permission_outcome(Status.FAIL), _permission_outcome(Status.FAIL)])

        with caplog.at_level(logging.ERROR):
            verdict.announce()

        assert "Every check was denied" in caplog.text
        assert "Preflight checks failed" in caplog.text

    def test_announce_reports_success_when_no_absorbed_check_failed(self, caplog: pytest.LogCaptureFixture) -> None:
        """A verdict with no absorbed failure announces success."""
        verdict = Verdict()
        verdict.absorb([_permission_outcome(Status.PASS)])

        with caplog.at_level(logging.INFO):
            verdict.announce()

        assert "Preflight checks passed" in caplog.text


class TestReporting:
    @pytest.mark.parametrize(
        ("status", "level", "prefix"),
        [
            (Status.PASS, logging.INFO, "PASS"),
            (Status.UNKNOWN, logging.INFO, "UNKNOWN"),
            (Status.FAIL, logging.ERROR, "FAIL"),
        ],
    )
    def test_each_status_uses_its_protocol_prefix_and_severity(
        self, caplog: pytest.LogCaptureFixture, status: Status, level: int, prefix: str
    ) -> None:
        """Each check status is reported with its stable prefix and severity."""
        with caplog.at_level(level):
            report(CheckResult(status=status, message="probe"))

        assert f"{prefix}  probe" in caplog.text

    def test_warn_emits_the_warning_protocol_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        """Preflight warnings use the stable warning prefix."""
        with caplog.at_level(logging.WARNING):
            warn("foreign object")

        assert "WARN  foreign object" in caplog.text
