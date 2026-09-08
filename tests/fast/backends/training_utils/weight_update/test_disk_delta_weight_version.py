import pytest


def _helpers():
    pytest.importorskip("sglang")
    from miles.backends.training_utils.weight_update.protocols.delta import (
        _UNSET_WEIGHT_VERSION,
        _update_weight_version_if_unset,
    )

    return _UNSET_WEIGHT_VERSION, _update_weight_version_if_unset


class TestUpdateWeightVersionIfUnset:
    def test_publishes_the_version_to_every_cold_engine(self):
        """Paths whose transport cannot carry the version announce it to the engines itself."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset(
            [_Engine(unset_version, calls, name="a"), _Engine(unset_version, calls, name="b")],
            "ab12cd34-00000007",
        )

        assert calls == [("a", "ab12cd34-00000007", False), ("b", "ab12cd34-00000007", False)]

    def test_announcing_does_not_abort_running_requests(self):
        """Announcing a version changes no weights, so it must not cancel in-flight generation."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine(unset_version, calls)], "ab12cd34-00000007")

        assert [abort for _name, _version, abort in calls] == [False]

    def test_an_engine_that_never_reported_a_version_is_cold(self):
        """A missing version is as unset as the launch default."""
        _unset, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine(None, calls)], "ab12cd34-00000007")

        assert [version for _name, version, _abort in calls] == ["ab12cd34-00000007"]

    def test_leaves_an_engine_that_already_serves_a_known_version_alone(self):
        """After a trainer failover the engines still hold their weights; claiming a version would lie."""
        _unset, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine("ab12cd34-00000004", calls)], "ab12cd34-00000007")

        assert calls == []

    def test_announces_only_to_the_engines_that_are_still_cold(self):
        """Partial recovery leaves survivors holding real weights; only the replacements need telling."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []
        engines = [
            _Engine("ab12cd34-00000004", calls, name="warm"),
            _Engine(unset_version, calls, name="cold"),
            _Engine(None, calls, name="never_told"),
        ]

        update_weight_version_if_unset(engines, "ab12cd34-00000007")

        assert [name for name, _version, _abort in calls] == ["cold", "never_told"]

    def test_no_engines_is_a_noop(self):
        """An updater with no engines attached publishes nothing rather than failing."""
        _unset, update_weight_version_if_unset = _helpers()

        update_weight_version_if_unset([], "ab12cd34-00000007")


class _Engine:
    def __init__(self, reported_version, calls, *, name=""):
        self._reported_version = reported_version
        self._calls = calls
        self._name = name

    async def get_weight_version(self):
        return self._reported_version

    async def update_weight_version(self, weight_version, abort_all_requests=True):
        self._calls.append((self._name, weight_version, abort_all_requests))
