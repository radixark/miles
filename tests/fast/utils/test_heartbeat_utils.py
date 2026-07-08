import miles.utils.ft_utils.heartbeat_utils as heartbeat_utils
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat


class TestSimpleHeartbeatBumpCount:
    def test_initial_bump_count_is_zero(self):
        """A fresh SimpleHeartbeat reports a bump_count of 0."""
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().bump_count == 0

    def test_single_bump_increments_count(self):
        """One bump() raises bump_count from 0 to 1."""
        heartbeat = SimpleHeartbeat()
        heartbeat.bump()
        assert heartbeat.status().bump_count == 1

    def test_bump_count_increments_monotonically(self):
        """Repeated bump() calls increment bump_count by exactly one each time."""
        heartbeat = SimpleHeartbeat()
        for expected in range(1, 6):
            heartbeat.bump()
            assert heartbeat.status().bump_count == expected

    def test_bump_count_never_resets(self):
        """bump_count keeps growing and never drops back across many bumps."""
        heartbeat = SimpleHeartbeat()
        previous = heartbeat.status().bump_count
        for _ in range(20):
            heartbeat.bump()
            current = heartbeat.status().bump_count
            assert current == previous + 1
            previous = current
        assert heartbeat.status().bump_count == 20


class TestSimpleHeartbeatStatusObject:
    def test_status_returns_heartbeat_status_instance(self):
        """status() yields a HeartbeatStatus dataclass instance."""
        heartbeat = SimpleHeartbeat()
        assert isinstance(heartbeat.status(), HeartbeatStatus)

    def test_status_is_frozen(self):
        """The returned HeartbeatStatus is frozen and rejects attribute mutation."""
        import dataclasses

        heartbeat = SimpleHeartbeat()
        status = heartbeat.status()
        try:
            status.bump_count = 999
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("expected FrozenInstanceError on frozen dataclass")

    def test_status_returns_new_object_each_bump(self):
        """Each bump() replaces the status with a distinct object identity."""
        heartbeat = SimpleHeartbeat()
        before = heartbeat.status()
        heartbeat.bump()
        after = heartbeat.status()
        assert before is not after

    def test_status_stable_between_bumps(self):
        """status() returns the same object identity when no bump occurs in between."""
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status() is heartbeat.status()

    def test_old_status_snapshot_unaffected_by_later_bump(self):
        """A captured status snapshot keeps its bump_count after subsequent bumps."""
        heartbeat = SimpleHeartbeat()
        snapshot = heartbeat.status()
        heartbeat.bump()
        heartbeat.bump()
        assert snapshot.bump_count == 0
        assert heartbeat.status().bump_count == 2


class TestSimpleHeartbeatTimestamp:
    def test_initial_timestamp_from_time_time(self, monkeypatch):
        """Construction stamps last_active_timestamp from time.time()."""
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: 1000.0)
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().last_active_timestamp == 1000.0

    def test_bump_updates_timestamp_from_time_time(self, monkeypatch):
        """bump() refreshes last_active_timestamp to the current time.time() value."""
        clock = {"now": 1000.0}
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: clock["now"])
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().last_active_timestamp == 1000.0

        clock["now"] = 2500.0
        heartbeat.bump()
        status = heartbeat.status()
        assert status.last_active_timestamp == 2500.0
        assert status.bump_count == 1

    def test_consecutive_bumps_track_timestamp(self, monkeypatch):
        """Each bump() captures the time.time() value observed at that call."""
        clock = {"now": 100.0}
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: clock["now"])
        heartbeat = SimpleHeartbeat()

        clock["now"] = 200.0
        heartbeat.bump()
        assert heartbeat.status().last_active_timestamp == 200.0

        clock["now"] = 350.0
        heartbeat.bump()
        assert heartbeat.status().last_active_timestamp == 350.0
        assert heartbeat.status().bump_count == 2
