from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.uninstall_lock import uninstall_lock


def test_same_generation_excludes_competitors_and_releases_after_failure(tmp_path: Path) -> None:
    """A failed owner releases the generation lock without letting a concurrent owner enter."""
    state_file = tmp_path / "generation.state"
    with pytest.raises(ValueError):
        with uninstall_lock(state_file):
            with pytest.raises(TimeoutError):
                with uninstall_lock(state_file, timeout_seconds=0):
                    pytest.fail("Competing owner entered")
            with uninstall_lock(tmp_path / "other.state", timeout_seconds=0):
                pass
            raise ValueError("owner failed")
    with uninstall_lock(state_file, timeout_seconds=0):
        pass
