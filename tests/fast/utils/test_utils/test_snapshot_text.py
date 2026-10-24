from pathlib import Path
import pytest
from miles.utils.test_utils.snapshot import SNAPSHOT_UPDATE_ENV_VAR, assert_matches_snapshot


class TestSnapshotFiles:

    def test_missing_baseline_reports_actual_content_without_writing_it(self, tmp_path: Path) -> None:
        """A failed assertion provides reviewable actual data and leaves baseline creation explicit."""
        path = tmp_path / 'missing.yaml'
        with pytest.raises(AssertionError) as error:
            assert_matches_snapshot(snapshot=path, actual='actual\n', subject='subject', update=False)
        assert 'BEGIN ACTUAL missing.yaml' in str(error.value)
        assert 'actual\n' in str(error.value)
        assert not path.exists()

    def test_explicit_read_only_mode_overrides_the_update_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Callers disabling updates cannot overwrite goldens through an inherited environment flag."""
        path = tmp_path / 'value.yaml'
        path.write_text('old\n')
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, '1')
        with pytest.raises(AssertionError):
            assert_matches_snapshot(snapshot=path, actual='new\n', subject='subject', update=False)
        assert path.read_text() == 'old\n'

    def test_explicit_update_creates_parent_directories(self, tmp_path: Path) -> None:
        """Approved baseline creation works for a new nested scenario directory."""
        path = tmp_path / 'nested' / 'value.yaml'
        assert_matches_snapshot(snapshot=path, actual='new\n', subject='subject', update=True)
        assert path.read_text() == 'new\n'
