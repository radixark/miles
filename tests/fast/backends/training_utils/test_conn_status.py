import pytest

from miles.backends.training_utils.conn_status import ConnStatusManager

_INITIAL_SNAPSHOT: dict[str, str] = {"cell-a": "hash-a", "cell-b": "hash-b"}


@pytest.mark.parametrize(
    ("next_snapshot", "expected_needs_reconnect"),
    [
        pytest.param(dict(_INITIAL_SNAPSHOT), False, id="unchanged"),
        pytest.param({**_INITIAL_SNAPSHOT, "cell-c": "hash-c"}, True, id="cell-added"),
        pytest.param({"cell-a": "hash-a"}, True, id="cell-removed"),
        pytest.param({"cell-a": "hash-a", "cell-b": "hash-b-restarted"}, True, id="cell-replaced"),
        pytest.param({"cell-a": "hash-a", "cell-c": "hash-b"}, True, id="cell-id-swapped"),
        pytest.param({}, True, id="all-cells-gone"),
    ],
)
def test_needs_reconnect_tracks_snapshot_changes(
    next_snapshot: dict[str, str], expected_needs_reconnect: bool
) -> None:
    """After a successful connect, only a changed rollout cell snapshot forces another reconnect."""
    manager = ConnStatusManager()
    manager.mark_reconnected(_INITIAL_SNAPSHOT)

    assert manager.needs_reconnect(next_snapshot) is expected_needs_reconnect


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param({}, id="empty-snapshot"),
        pytest.param({"cell-a": "hash-a"}, id="non-empty-snapshot"),
    ],
)
def test_needs_reconnect_is_true_before_first_connect(snapshot: dict[str, str]) -> None:
    """A freshly created manager always reconnects, including for a defensive empty snapshot."""
    manager = ConnStatusManager()

    assert manager.needs_reconnect(snapshot) is True


def test_empty_snapshot_stops_reconnecting_once_marked() -> None:
    """An empty snapshot is a real state: after marking it connected, an equal empty snapshot needs no reconnect."""
    manager = ConnStatusManager()
    manager.mark_reconnected({})

    assert manager.needs_reconnect({}) is False


def test_trainer_stale_forces_reconnect_on_unchanged_snapshot() -> None:
    """A stale trainer reconnects even when the rollout cell snapshot is identical, and clears the flag afterwards."""
    manager = ConnStatusManager()
    manager.mark_reconnected(_INITIAL_SNAPSHOT)
    manager.mark_trainer_stale()

    assert manager.needs_reconnect(dict(_INITIAL_SNAPSHOT)) is True

    manager.mark_reconnected(_INITIAL_SNAPSHOT)

    assert manager.needs_reconnect(dict(_INITIAL_SNAPSHOT)) is False


def test_reconnect_decisions_are_repeatable_across_updates() -> None:
    """Scale-down then scale-up each trigger exactly one reconnect, and the steady state stays connected."""
    manager = ConnStatusManager()
    scaled_up: dict[str, str] = {"cell-a": "hash-a", "cell-b": "hash-b"}
    scaled_down: dict[str, str] = {"cell-a": "hash-a"}

    manager.mark_reconnected(scaled_up)

    assert manager.needs_reconnect(scaled_down) is True

    manager.mark_reconnected(scaled_down)

    assert manager.needs_reconnect(scaled_down) is False
    assert manager.needs_reconnect(scaled_up) is True


def test_stored_snapshot_is_isolated_from_caller_mutation() -> None:
    """The manager copies the snapshot it was given, so mutating the caller's dict still reads as a change."""
    manager = ConnStatusManager()
    snapshot: dict[str, str] = {"cell-a": "hash-a"}
    manager.mark_reconnected(snapshot)

    snapshot["cell-a"] = "hash-a-restarted"

    assert manager.needs_reconnect(snapshot) is True
