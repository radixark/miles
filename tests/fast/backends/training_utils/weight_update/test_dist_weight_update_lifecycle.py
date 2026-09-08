import asyncio
import concurrent.futures
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


def _make_updater(engines: list[_RecordingApiClient], *, pause_generation_mode: str = "retract") -> WeightUpdater:
    protocol = SimpleNamespace(
        use_weight_update_session=True,
        needs_base_resync_for_lora=False,
        is_sender=True,
        group_name="test",
        rollout_engines=engines,
        required_placement=MagicMock(),
        supports_lora=False,
        begin_sync=lambda weight_version, iter_buckets: True,
        send_bucket=MagicMock(),
        after_base_weights=MagicMock(),
        finalize=MagicMock(),
    )
    iterator = MagicMock()
    iterator.iter_hf_weights.return_value = iter([])
    iterator.weight_update_selector = "all"
    args = Namespace(pause_generation_mode=pause_generation_mode, check_lora_weight_equal=False)
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


def _run(updater: WeightUpdater, *, rank: int = 0) -> None:
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = rank
        updater.update_weights()


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

    def test_a_failed_pause_opens_no_update_session(self):
        """A session opened on an engine that never paused would load weights under generation."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="pause_generation"))

        with pytest.raises(RuntimeError, match="pause_generation failed"):
            _run(updater)

        assert _phases(calls) == ["pause_generation"]
        assert _engines_called(calls, "pause_generation") == [1]

    def test_a_failed_flush_opens_no_update_session(self):
        """A session opened over a stale cache would serve tokens generated from the old weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="flush_cache"))

        with pytest.raises(RuntimeError, match="flush_cache failed"):
            _run(updater)

        assert _phases(calls) == ["pause_generation", "flush_cache"]
        assert _engines_called(calls, "flush_cache") == [1]

    def test_a_failed_begin_prevents_the_update_from_starting(self):
        """A begin failure must escape instead of letting weight transfer proceed with a closed engine."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="begin_weight_update", failing_engine_index=1))

        with pytest.raises(RuntimeError, match="begin_weight_update failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES
        assert _engines_called(calls, "begin_weight_update") == [0]
        updater.protocol.send_bucket.assert_not_called()

    def test_a_failed_session_close_neither_publishes_the_version_nor_resumes(self):
        """An engine that resumed without a post-load pass would serve packed weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="end_weight_update"))

        with pytest.raises(RuntimeError, match="end_weight_update failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES + ["end_weight_update"]
        assert _engines_called(calls, "end_weight_update") == [1]

    def test_a_failed_version_publication_does_not_resume(self):
        """An engine resuming under a version it never acknowledged would mislabel its samples."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="update_weight_version"))

        with pytest.raises(RuntimeError, match="update_weight_version failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES + ["end_weight_update", "update_weight_version"]
        assert _engines_called(calls, "update_weight_version") == [1]

    def test_non_source_rank_issues_no_requests(self):
        """Every rank runs the updater, but only rank 0 may drive the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))

        _run(updater, rank=1)

        assert calls == []
