import json
import logging
from pathlib import Path

import pytest

from miles.utils.env_report.launcher_report import read_launcher_report


class TestReadLauncherReport:
    def test_reads_the_record_the_launcher_wrote(self, tmp_path: Path) -> None:
        """The launcher hands a path rather than the record itself, so the value has to be opened."""
        path = tmp_path / "launch-1.json"
        path.write_text(json.dumps({"run_id": "260101-000000-000", "release": "miles-run"}))

        assert read_launcher_report(str(path)) == {"run_id": "260101-000000-000", "release": "miles-run"}

    def test_returns_none_when_no_launcher_named_a_record(self) -> None:
        """A run started by hand has no launcher and no record, which is not a failure."""
        assert read_launcher_report("") is None

    def test_a_missing_record_only_warns(self, tmp_path: Path, caplog) -> None:
        """A shared disk that lost the record must not stop the process it describes from starting."""
        with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.launcher_report"):
            assert read_launcher_report(str(tmp_path / "gone.json")) is None

        assert "Failed to read the launcher report" in caplog.text
        assert caplog.records[0].exc_info is not None

    def test_a_corrupt_record_only_warns(self, tmp_path: Path, caplog) -> None:
        """A half written record is a broken launch, not a reason to refuse to run."""
        path = tmp_path / "launch-1.json"
        path.write_text("{not json at all")

        with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.launcher_report"):
            assert read_launcher_report(str(path)) is None

    @pytest.mark.parametrize("raw", ["[]", '"text"', "3"])
    def test_returns_none_for_json_that_is_not_an_object(self, raw: str, tmp_path: Path) -> None:
        """The report is a mapping; handing anything else to the model would fail the process's startup."""
        path = tmp_path / "launch-1.json"
        path.write_text(raw)

        assert read_launcher_report(str(path)) is None
