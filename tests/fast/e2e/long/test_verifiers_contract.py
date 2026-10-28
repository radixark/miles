from pathlib import Path

import pytest
from tests.e2e.long.verifiers_contract import _assert_contract_report


class TestVerifiersSdkContractReport:
    def test_three_completed_cases_are_accepted(self, tmp_path: Path) -> None:
        """All three real SDK witnesses must complete successfully."""
        report = tmp_path / "report.xml"
        report.write_text("<testsuites><testsuite>" + "<testcase/>" * 3 + "</testsuite></testsuites>")

        _assert_contract_report(report)

    @pytest.mark.parametrize("status", ["skipped", "failure", "error"])
    def test_incomplete_or_unsuccessful_cases_are_rejected(self, tmp_path: Path, status: str) -> None:
        """A zero pytest exit code cannot hide a skipped SDK witness."""
        report = tmp_path / "report.xml"
        report.write_text(f"<testsuite><testcase/><testcase/><testcase><{status}/></testcase></testsuite>")

        with pytest.raises(AssertionError, match="did not pass"):
            _assert_contract_report(report)

    @pytest.mark.parametrize("count", [0, 2, 4])
    def test_missing_or_extra_cases_are_rejected(self, tmp_path: Path, count: int) -> None:
        """A changed collection cannot silently remove a required SDK witness."""
        report = tmp_path / "report.xml"
        report.write_text("<testsuite>" + "<testcase/>" * count + "</testsuite>")

        with pytest.raises(AssertionError, match="Expected three"):
            _assert_contract_report(report)
