from miles.backends.training_utils.conn_status import ConnStatusManager


def test_fresh_manager_requires_an_initial_connect() -> None:
    """A never-connected manager reports that a reconnect is still required."""
    manager = ConnStatusManager()

    assert manager.needs_reconnect() is True


def test_reconnect_is_not_repeated_after_a_successful_connect() -> None:
    """Once marked reconnected, the manager stops asking for further reconnects."""
    manager = ConnStatusManager()

    manager.mark_reconnected()

    assert manager.needs_reconnect() is False


def test_marking_the_trainer_stale_forces_another_reconnect() -> None:
    """A stale trainer makes an already-connected manager require a reconnect again."""
    manager = ConnStatusManager()
    manager.mark_reconnected()

    manager.mark_trainer_stale()

    assert manager.needs_reconnect() is True


def test_reconnecting_clears_the_stale_trainer_flag() -> None:
    """Reconnecting after a stale trainer clears the flag so later windows skip reconnects."""
    manager = ConnStatusManager()
    manager.mark_reconnected()
    manager.mark_trainer_stale()

    manager.mark_reconnected()

    assert manager.needs_reconnect() is False
    assert manager.needs_reconnect() is False


def test_marking_the_trainer_stale_before_any_connect_still_requires_reconnect() -> None:
    """Staleness on a never-connected manager keeps the reconnect requirement set."""
    manager = ConnStatusManager()

    manager.mark_trainer_stale()

    assert manager.needs_reconnect() is True
