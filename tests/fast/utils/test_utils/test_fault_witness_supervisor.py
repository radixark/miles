from pathlib import Path

import pytest

from miles.utils.test_utils import fault_witness_supervisor
from miles.utils.test_utils.fault_witness import WITNESS_DIRECTORY_ENV, _witness_lease


@pytest.mark.parametrize("publication_failed", [False, True])
def test_supervisor_waits_for_live_witness_and_releases_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication_failed: bool
) -> None:
    """The container cannot exit with a live witness, and failed publication releases its lease."""
    monkeypatch.setenv(WITNESS_DIRECTORY_ENV, str(tmp_path))
    times = iter([0.0, 26.0])
    monkeypatch.setattr(fault_witness_supervisor.time, "monotonic", lambda: next(times))
    try:
        with _witness_lease():
            with pytest.raises(TimeoutError, match="Fault witnesses"):
                fault_witness_supervisor._drain_witnesses(tmp_path)
            if publication_failed:
                raise ConnectionError("Receipt destination unavailable")
    except ConnectionError:
        assert publication_failed
    monkeypatch.setattr(fault_witness_supervisor.time, "monotonic", lambda: 0.0)
    fault_witness_supervisor._drain_witnesses(tmp_path)


def test_no_fault_does_not_delay_container_exit(tmp_path: Path) -> None:
    """Ordinary completion has no witness lease to keep the container alive."""
    fault_witness_supervisor._drain_witnesses(tmp_path)
