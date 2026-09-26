import asyncio
import concurrent.futures
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils import async_utils

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_SESSION_MODULE = "miles.backends.training_utils.weight_update.session"

_ENGINE_COUNT = 2
_PREPARE_PHASES = ["pause_generation", "flush_cache", "begin_weight_update"]
_FINALIZE_PHASES = ["end_weight_update", "update_weight_version", "continue_generation"]


class _ObservedFuture:
    def __init__(self, future: concurrent.futures.Future, result_started: threading.Event):
        self._future = future
        self._result_started = result_started

    def result(self):
        self._result_started.set()
        return self._future.result()


class _RecordingApiClient:
    def __init__(
        self,
        calls: list[tuple[int, str, dict]],
        engine_index: int,
        failing_method: str | None = None,
        gates: dict[str, threading.Event] | None = None,
    ):
        self._calls = calls
        self._engine_index = engine_index
        self._failing_method = failing_method
        self._gates = gates if gates is not None else {}

    def __getattr__(self, name: str):
        async def method(**kwargs):
            if (gate := self._gates.get(name)) is not None and not await asyncio.to_thread(gate.wait, 5):
                raise TimeoutError(f"{name} gate timed out")
            if name == self._failing_method:
                raise RuntimeError(f"{name} failed")
            self._calls.append((self._engine_index, name, kwargs))
            return {"success": True}

        return method


def _make_engines(
    calls: list[tuple[int, str, dict]],
    *,
    failing_method: str | None = None,
    failing_engine_index: int = 0,
    later_engine_gates: dict[str, threading.Event] | None = None,
) -> list[_RecordingApiClient]:
    return [
        _RecordingApiClient(
            calls,
            engine_index,
            failing_method if engine_index == failing_engine_index else None,
            later_engine_gates if engine_index == _ENGINE_COUNT - 1 else None,
        )
        for engine_index in range(_ENGINE_COUNT)
    ]


def _make_updater(
    engines: list[_RecordingApiClient], *, pause_generation_mode: str = "retract", begin_sync_result: bool = True
) -> WeightUpdater:
    begin_sync_versions: list[int] = []

    def begin_sync(weight_version: int, iter_buckets) -> bool:
        begin_sync_versions.append(weight_version)
        return begin_sync_result

    protocol = SimpleNamespace(
        use_weight_update_session=True,
        needs_base_resync_for_lora=False,
        is_sender=True,
        group_name="test",
        rollout_engines=engines,
        cell_updaters_of_cell_id=create_rollout_cell_updaters(
            args=Namespace(update_weight_engine_request_timeout=10.0),
            rollout_engines=engines,
            engine_cell_ids=[f"cell-{index}" for index in range(len(engines))],
        ),
        required_placement=MagicMock(),
        supports_lora=False,
        begin_sync=begin_sync,
        begin_sync_versions=begin_sync_versions,
        send_bucket=MagicMock(),
        after_base_weights=MagicMock(),
        finalize=MagicMock(),
        after_engines_resumed=MagicMock(),
    )
    iterator = MagicMock()
    iterator.iter_hf_weights.return_value = iter([])
    iterator.weight_update_selector = "all"
    args = Namespace(
        pause_generation_mode=pause_generation_mode,
        check_lora_weight_equal=False,
        fully_async=False,
        colocate=True,
    )
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        return WeightUpdater(
            args,
            [MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            iterator_factory=lambda *a, **k: iterator,
            parallel_state=MagicMock(),
            is_lora=False,
        )


def _run(updater: WeightUpdater, *, rank: int = 0, weight_version: int = 1) -> None:
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = rank
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = lambda results, value, group: results.__setitem__(0, value)
        return updater.update_weights(weight_version=weight_version)


def _phases(calls: list[tuple[int, str, dict]]) -> list[str]:
    phases: list[str] = []
    for _engine_index, name, _kwargs in calls:
        if not phases or phases[-1] != name:
            phases.append(name)
    return phases


def _engines_called(calls: list[tuple[int, str, dict]], method: str) -> list[int]:
    return sorted(engine_index for engine_index, name, _kwargs in calls if name == method)


def _kwargs_of(calls: list[tuple[int, str, dict]], method: str) -> list[dict]:
    return [kwargs for _engine_index, name, kwargs in calls if name == method]


def _run_with_gated_later_engine(
    updater: WeightUpdater,
    calls: list[tuple[int, str, dict]],
    phases: list[str],
    gates: dict[str, threading.Event],
) -> None:
    result_started = {phase: threading.Event() for phase in phases}
    submission_count = 0
    original_submit = async_utils.submit

    def observed_submit(coro):
        nonlocal submission_count
        future = original_submit(coro)
        submission_count += 1
        if submission_count % _ENGINE_COUNT == 0:
            phase = phases[submission_count // _ENGINE_COUNT - 1]
            return _ObservedFuture(future, result_started[phase])
        return future

    with patch(f"{_SESSION_MODULE}.async_utils.submit", side_effect=observed_submit):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = executor.submit(_run, updater)
            try:
                for phase_index, phase in enumerate(phases):
                    assert result_started[phase].wait(timeout=5)
                    assert not outcome.done()
                    assert not any(name in phases[phase_index + 1 :] for _engine_index, name, _kwargs in calls)
                    gates[phase].set()
                outcome.result(timeout=5)
            finally:
                for gate in gates.values():
                    gate.set()

    assert submission_count == len(phases) * _ENGINE_COUNT


class TestWeightUpdateSessionFrame:
    """The session frame must pause, flush, open, close, publish the version and only then resume, on every engine."""

    @pytest.mark.parametrize("pause_generation_mode", ["retract", "abort"])
    def test_every_engine_walks_the_frame_in_order(self, pause_generation_mode):
        """Loading weights into a still-generating engine, or resuming a half-closed one, corrupts the rollout."""
        calls: list[tuple[int, str, dict]] = []
        phases = _PREPARE_PHASES + _FINALIZE_PHASES
        gates = {phase: threading.Event() for phase in phases}
        updater = _make_updater(
            _make_engines(calls, later_engine_gates=gates), pause_generation_mode=pause_generation_mode
        )

        _run_with_gated_later_engine(updater, calls, phases, gates)

        assert _phases(calls) == phases
        for method in phases:
            assert _engines_called(calls, method) == list(range(_ENGINE_COUNT))
        assert _kwargs_of(calls, "pause_generation") == [{"mode": pause_generation_mode}] * _ENGINE_COUNT
        assert _kwargs_of(calls, "begin_weight_update") == [{"selector": "all", "sync_base": True}] * _ENGINE_COUNT
        assert _kwargs_of(calls, "update_weight_version") == [{"weight_version": "1"}] * _ENGINE_COUNT

    def test_in_place_pause_mode_skips_the_flush(self):
        """in_place pause keeps the running requests, so their cache must survive."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls), pause_generation_mode="in_place")

        _run(updater)

        assert _phases(calls) == ["pause_generation", "begin_weight_update"] + _FINALIZE_PHASES
        assert _kwargs_of(calls, "pause_generation") == [{"mode": "in_place"}] * _ENGINE_COUNT

    @pytest.mark.parametrize("failed_method", _PREPARE_PHASES + _FINALIZE_PHASES[:-1])
    @pytest.mark.parametrize("failed_engine_index", [0, 1])
    def test_a_failed_cell_stops_while_healthy_cells_complete_the_frame(
        self, failed_method: str, failed_engine_index: int
    ) -> None:
        """A broken cell never advances past its failed phase while healthy cells publish and resume."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(
            _make_engines(calls, failing_method=failed_method, failing_engine_index=failed_engine_index)
        )

        _run(updater)

        phases = _PREPARE_PHASES + _FINALIZE_PHASES
        for engine_index in range(_ENGINE_COUNT):
            cell = updater.protocol.cell_updaters_of_cell_id[f"cell-{engine_index}"]
            observed = [name for index, name, _kwargs in calls if index == engine_index]
            if engine_index == failed_engine_index:
                assert cell.is_errored
                assert str(cell._error) == f"{failed_method} failed"
                assert observed == phases[: phases.index(failed_method)]
            else:
                assert not cell.is_errored
                assert observed == phases

    def test_a_trainer_protocol_failure_still_aborts_before_engine_publication(self) -> None:
        """A trainer-wide transfer failure must escape without publishing or resuming any engine."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))
        updater.protocol.finalize.side_effect = RuntimeError("trainer transfer failed")

        with pytest.raises(RuntimeError, match="trainer transfer failed"):
            _run(updater)

        for engine_index in range(_ENGINE_COUNT):
            assert [name for index, name, _kwargs in calls if index == engine_index] == _PREPARE_PHASES

    def test_non_source_rank_issues_no_requests(self):
        """Every rank runs the updater, but only rank 0 may drive the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))

        _run(updater, rank=1)

        assert calls == []


class TestExplicitWeightVersion:
    """The updater publishes the ordinal it is handed instead of counting locally."""

    def test_the_assigned_ordinal_reaches_the_protocol_and_the_engines(self):
        """A cell healed mid-run starts from a zeroed cache, so counting locally would republish an old version."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))

        result = _run(updater, weight_version=7)

        assert result is None
        assert "weight_version" not in vars(updater)
        assert updater.protocol.begin_sync_versions == [7]
        updater.protocol.finalize.assert_called_once_with(7)
        assert _kwargs_of(calls, "update_weight_version") == [{"weight_version": "7"}] * _ENGINE_COUNT

    def test_consecutive_updates_follow_the_assigned_ordinals(self):
        """Two syncs in a row must carry exactly the two ordinals the controller reserved."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))

        _run(updater, weight_version=4)
        _run(updater, weight_version=5)

        assert updater.protocol.begin_sync_versions == [4, 5]
        assert (
            _kwargs_of(calls, "update_weight_version")
            == [{"weight_version": "4"}] * _ENGINE_COUNT + [{"weight_version": "5"}] * _ENGINE_COUNT
        )

    def test_skipped_sync_returns_none_without_contacting_engines(self):
        """A skipped transfer cannot claim that its requested version reached the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls), begin_sync_result=False)

        result = _run(updater, weight_version=1)

        assert result is None
        assert calls == []
