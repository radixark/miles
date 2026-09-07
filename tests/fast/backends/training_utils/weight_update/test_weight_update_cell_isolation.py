from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_CELL_IDS = ["cell-0", "cell-1"]


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, str]], cell_id: str, failing_method: str | None = None):
        self._calls = calls
        self._cell_id = cell_id
        self._failing_method = failing_method

    def __getattr__(self, name: str):
        async def method(**_kwargs):
            self._calls.append((self._cell_id, name))
            if name == self._failing_method:
                raise ConnectionError(f"{self._cell_id} is unreachable")
            return {"success": True}

        return method


class _FakeCellIsolatingProtocol:
    use_weight_update_session = True
    supports_lora = False
    needs_base_resync_for_lora = False

    def __init__(self) -> None:
        self.args = None
        self.required_placement = MagicMock()
        self.inference_cell_health = InferenceCellHealth()
        self.is_sender = True
        self.rollout_engines: list = []
        self.group_name = "test"
        self.sent_buckets: list[list] = []

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        engine_cell_ids,
        parallel_state,
        placement,
        selector,
    ) -> None:
        self.rollout_engines = list(rollout_engines)
        self.inference_cell_health = InferenceCellHealth(engine_cell_ids)

    def begin_sync(self, weight_version, iter_buckets) -> bool:
        return True

    def send_bucket(self, bucket) -> None:
        self.sent_buckets.append(list(bucket))

    def after_base_weights(self) -> None:
        pass

    def finalize(self, weight_version) -> None:
        pass


def _make_updater(engines: list[_RecordingApiClient], protocol: _FakeCellIsolatingProtocol) -> WeightUpdater:
    iterator = MagicMock()
    iterator.iter_hf_weights.return_value = iter([])
    iterator.weight_update_selector = "all"
    args = Namespace(pause_generation_mode="retract", check_lora_weight_equal=False, update_weight_transfer_mode="p2p")

    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        updater = WeightUpdater(
            args,
            [MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            iterator_factory=lambda *_args, **_kwargs: iterator,
            parallel_state=MagicMock(),
            is_lora=False,
        )
    updater.connect_rollout_engines(engines, [1, 1], [0, 1], engine_cell_ids=_CELL_IDS)
    return updater


def _run(updater: WeightUpdater, *, rank: int = 0, weight_version: int = 1) -> int:
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = rank
        return updater.update_weights(weight_version=weight_version)


class TestPerCellSessionFrame:
    """The driver runs the engine session of every assigned cell independently."""

    def test_every_healthy_cell_walks_the_whole_frame(self) -> None:
        """The per-cell path replaces the fleet-wide one, so it must still open and close every session."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)

        _run(updater)

        assert [name for cell_id, name in calls if cell_id == "cell-0"] == [
            "pause_generation",
            "flush_cache",
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
            "continue_generation",
        ]
        assert sorted({cell_id for cell_id, _name in calls}) == _CELL_IDS

    def test_a_cell_that_fails_to_begin_is_never_resumed(self) -> None:
        """Resuming an engine that missed the update lets it serve the previous weights as if they were current."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        engines = [
            _RecordingApiClient(calls, "cell-0", failing_method="begin_weight_update"),
            _RecordingApiClient(calls, "cell-1"),
        ]
        updater = _make_updater(engines, protocol)

        _run(updater)

        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]
        assert ("cell-0", "continue_generation") not in calls
        assert ("cell-0", "update_weight_version") not in calls
        assert ("cell-1", "continue_generation") in calls

    def test_a_non_driver_rank_drives_no_engine(self) -> None:
        """Only one rank owns the session frame; a second driver would pause an engine mid-write."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)

        _run(updater, rank=1)

        assert calls == []

    def test_a_failed_cell_does_not_fail_the_update(self) -> None:
        """The trainer keeps its state and its healthy targets; the failed cell is reported, not raised."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        engines = [
            _RecordingApiClient(calls, "cell-0", failing_method="pause_generation"),
            _RecordingApiClient(calls, "cell-1"),
        ]
        updater = _make_updater(engines, protocol)

        assert _run(updater, weight_version=4) == 4
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]
        assert protocol.inference_cell_health.healthy_cell_ids == ["cell-1"]
