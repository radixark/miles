from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_HEALTH_MODULE = "miles.backends.training_utils.weight_update.inference_cell_health"
_CELL_IDS = ["cell-0", "cell-1"]


def _patched_health_dist(other_rank_reports: list[list[str]]):
    def all_gather_object(gathered, obj, group=None):
        gathered[0] = list(obj)
        for index, report in enumerate(other_rank_reports):
            gathered[index + 1] = list(report)

    dist_mock = MagicMock()
    dist_mock.get_world_size.return_value = 1 + len(other_rank_reports)
    dist_mock.all_gather_object.side_effect = all_gather_object
    return patch(f"{_HEALTH_MODULE}.dist", dist_mock)


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


def _run(
    updater: WeightUpdater,
    *,
    rank: int = 0,
    weight_version: int = 1,
    other_rank_reports: list[list[str]] | None = None,
) -> int:
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
        _patched_health_dist(other_rank_reports if other_rank_reports is not None else []),
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


class TestCrossRankAgreement:
    """Only one rank has to fail to write a cell for the whole trainer cell to give up on it."""

    def test_a_cell_another_rank_lost_is_never_resumed_here(self) -> None:
        """A cell that missed one shard serves a mixed model, so resuming it hides the corruption."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)

        _run(updater, other_rank_reports=[["cell-0"]])

        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]
        assert ("cell-0", "end_weight_update") not in calls
        assert ("cell-0", "update_weight_version") not in calls
        assert ("cell-0", "continue_generation") not in calls
        assert ("cell-1", "continue_generation") in calls

    def test_a_cell_this_rank_lost_is_reported_to_the_others(self) -> None:
        """The other ranks keep writing to a cell nobody told them about."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        engines = [
            _RecordingApiClient(calls, "cell-0", failing_method="begin_weight_update"),
            _RecordingApiClient(calls, "cell-1"),
        ]
        updater = _make_updater(engines, protocol)
        reported: list[list[str]] = []

        def all_gather_object(gathered, obj, group=None):
            reported.append(list(obj))
            gathered[0] = list(obj)
            gathered[1] = []

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_HEALTH_MODULE}.dist") as health_dist,
        ):
            dist_mock.get_rank.return_value = 0
            health_dist.get_world_size.return_value = 2
            health_dist.all_gather_object.side_effect = all_gather_object
            updater.update_weights(weight_version=1)

        assert reported == [["cell-0"], ["cell-0"], ["cell-0"]]

    def test_a_rank_that_lost_no_cell_still_joins_the_aggregation(self) -> None:
        """A collective inside a rank-conditional branch would hang the ranks that skipped it."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            _patched_health_dist([["cell-1"]]) as health_dist,
        ):
            dist_mock.get_rank.return_value = 3
            updater.update_weights(weight_version=1)

        assert health_dist.all_gather_object.call_count == 4
        assert calls == []
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-1"]

    def test_a_resume_that_fails_reaches_every_rank(self) -> None:
        """op12 exports one report per update, and a cell resumed into a stale model must not be in it as healthy."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        engines = [
            _RecordingApiClient(calls, "cell-0", failing_method="continue_generation"),
            _RecordingApiClient(calls, "cell-1"),
        ]
        updater = _make_updater(engines, protocol)
        reported: list[list[str]] = []

        def all_gather_object(gathered, obj, group=None):
            reported.append(list(obj))
            gathered[0] = list(obj)
            gathered[1] = []

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_HEALTH_MODULE}.dist") as health_dist,
        ):
            dist_mock.get_rank.return_value = 0
            health_dist.get_world_size.return_value = 2
            health_dist.all_gather_object.side_effect = all_gather_object
            updater.update_weights(weight_version=1)

        assert ("cell-0", "continue_generation") in calls
        assert reported[-1] == ["cell-0"]
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]
        assert ("cell-1", "continue_generation") in calls

    def test_a_rank_that_drives_no_engine_learns_of_a_failed_resume(self) -> None:
        """A non-driver rank never sees the resume request, so only the aggregation can tell it the cell is gone."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)
        syncs: list[list[str]] = []

        def all_gather_object(gathered, obj, group=None):
            syncs.append(list(obj))
            gathered[0] = list(obj)
            gathered[1] = ["cell-0"] if len(syncs) >= 4 else []

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_HEALTH_MODULE}.dist") as health_dist,
        ):
            dist_mock.get_rank.return_value = 2
            health_dist.get_world_size.return_value = 2
            health_dist.all_gather_object.side_effect = all_gather_object
            updater.update_weights(weight_version=1)

        assert len(syncs) == 4
        assert calls == []
        assert protocol.inference_cell_health.errored_cell_ids == ["cell-0"]

    def test_a_rank_without_a_healthy_target_still_joins_the_final_aggregation(self) -> None:
        """Skipping the last collective on a rank whose cells all failed would hang the ranks that took it."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        engines = [_RecordingApiClient(calls, cell_id, failing_method="pause_generation") for cell_id in _CELL_IDS]
        updater = _make_updater(engines, protocol)

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            _patched_health_dist([[]]) as health_dist,
        ):
            dist_mock.get_rank.return_value = 0
            updater.update_weights(weight_version=1)

        assert health_dist.all_gather_object.call_count == 4
        assert protocol.inference_cell_health.healthy_cell_ids == []

    def test_the_failures_are_aggregated_before_the_engines_are_resumed(self) -> None:
        """A verdict that lands after the resume request cannot take it back."""
        calls: list[tuple[str, str]] = []
        protocol = _FakeCellIsolatingProtocol()
        updater = _make_updater([_RecordingApiClient(calls, cell_id) for cell_id in _CELL_IDS], protocol)
        gathered_at: list[list[tuple[str, str]]] = []

        def all_gather_object(gathered, obj, group=None):
            gathered_at.append(list(calls))
            gathered[0] = list(obj)
            gathered[1] = ["cell-0"] if len(gathered_at) >= 3 else []

        with (
            patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_HEALTH_MODULE}.dist") as health_dist,
        ):
            dist_mock.get_rank.return_value = 0
            health_dist.get_world_size.return_value = 2
            health_dist.all_gather_object.side_effect = all_gather_object
            updater.update_weights(weight_version=1)

        assert len(gathered_at) == 4
        assert ("cell-0", "update_weight_version") in gathered_at[2]
        assert not [entry for entry in gathered_at[2] if entry[1] == "continue_generation"]
        assert ("cell-0", "continue_generation") not in calls
        assert ("cell-1", "continue_generation") in calls
