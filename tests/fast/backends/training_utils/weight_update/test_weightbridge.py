from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.protocols import weightbridge as weightbridge_module
from miles.backends.training_utils.weight_update.protocols.weightbridge import UpdateWeightWeightBridge
from miles.backends.training_utils.weight_update.updater import WeightUpdater


class _RecordingAdapter:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def send_weights(self, metadata=None):
        self._calls.append(("send_weights", metadata))
        return SimpleNamespace(synchronize=lambda: self._calls.append(("pack_event_synchronize", None)))

    def wait_send_complete(self) -> None:
        self._calls.append(("wait_send_complete", None))

    def flush_profile_outputs(self) -> None:
        self._calls.append(("flush_profile_outputs", None))


def _protocol(*, async_delivery: bool, calls: list) -> UpdateWeightWeightBridge:
    protocol = UpdateWeightWeightBridge(Namespace(weightbridge_async_delivery=async_delivery))
    protocol._adapter = _RecordingAdapter(calls)
    return protocol


class TestSendLiveWeights:
    def test_sync_delivery_waits_for_every_receiver_and_tags_the_trainer_version(self) -> None:
        """The next rollout must run on the new weights, labelled with the trainer's own version."""
        calls: list = []
        protocol = _protocol(async_delivery=False, calls=calls)

        protocol.send_live_weights(weight_version=7)

        assert calls == [
            ("send_weights", {"weight_version": 7}),
            ("pack_event_synchronize", None),
            ("wait_send_complete", None),
            ("flush_profile_outputs", None),
        ]

    def test_async_delivery_returns_once_the_weights_are_packed(self) -> None:
        """With async delivery the trainer must not block on the engines loading the update."""
        calls: list = []
        protocol = _protocol(async_delivery=True, calls=calls)

        protocol.send_live_weights(weight_version=7)
        protocol.send_live_weights(weight_version=8)

        assert ("wait_send_complete", None) not in calls
        assert [metadata for name, metadata in calls if name == "send_weights"] == [
            {"weight_version": 7},
            {"weight_version": 8},
        ]


class TestConnect:
    @pytest.fixture
    def connected(self, monkeypatch: pytest.MonkeyPatch) -> UpdateWeightWeightBridge:
        monkeypatch.setattr(weightbridge_module, "_coordinator_url", lambda server_url: f"coord:{server_url}")
        protocol = UpdateWeightWeightBridge(Namespace(weightbridge_async_delivery=False))
        protocol._connected_urls = ("coord:http://a:1", "coord:http://b:2")
        protocol._build_sender_adapter = lambda receiver_urls: pytest.fail("connect must not rebuild the sender")
        return protocol

    @staticmethod
    def _connect(protocol: UpdateWeightWeightBridge, urls: list[str]) -> None:
        protocol.connect(
            rollout_engines=[SimpleNamespace(server_url=url) for url in urls],
            engine_gpu_counts=None,
            engine_gpu_offsets=None,
            engine_cell_ids=[f"cell-{i}" for i in range(len(urls))],
            parallel_state=None,
            placement=None,
            selector="all",
        )

    def test_reconnecting_the_same_engines_keeps_the_one_time_connection(self, connected) -> None:
        """WeightBridge connects once; a repeated connect for the same engines must reuse it."""
        self._connect(connected, ["http://a:1", "http://b:2"])

        assert connected.is_sender is True
        assert set(connected.cell_updaters_of_cell_id) == {"cell-0", "cell-1"}

    def test_a_changed_engine_set_is_refused(self, connected) -> None:
        """Silently keeping the old connection would send weights to engines that are gone."""
        with pytest.raises(NotImplementedError, match="changed rollout engine set"):
            self._connect(connected, ["http://a:1", "http://c:3"])


def test_updater_hands_the_whole_update_to_a_non_streaming_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """WeightBridge owns the engine side: no session RPC and no HF bucket stream may run."""
    for name in ("maybe_pause_engines", "begin_weight_update", "end_weight_update", "set_weight_version"):
        monkeypatch.setattr(updater_module, name, lambda *args, _name=name, **kwargs: pytest.fail(f"{_name} called"))
    sent: list[int] = []
    updater = WeightUpdater.__new__(WeightUpdater)
    updater.protocol = SimpleNamespace(streams_buckets=False, send_live_weights=sent.append)
    updater._hf_weight_iterator = SimpleNamespace(
        iter_hf_weights=lambda *args, **kwargs: pytest.fail("HF buckets must not be built")
    )

    assert updater.update_weights(weight_version=12) is True
    assert sent == [12]
