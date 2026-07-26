import asyncio
import concurrent.futures
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
    DistBucketedWeightUpdateMixin,
)
from miles.utils import async_utils

_MIXIN_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"

_ENGINE_COUNT = 2


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


def _make_updater(
    calls: list[tuple[int, str, dict]],
    *,
    failing_method: str | None = None,
    failing_engine_index: int = 0,
    later_engine_gates: dict[str, threading.Event] | None = None,
    pause_generation_mode: str = "retract",
) -> SimpleNamespace:
    engines = [
        _RecordingApiClient(
            calls,
            engine_index,
            failing_method if engine_index == failing_engine_index else None,
            later_engine_gates if engine_index == _ENGINE_COUNT - 1 else None,
        )
        for engine_index in range(_ENGINE_COUNT)
    ]
    return SimpleNamespace(
        args=Namespace(pause_generation_mode=pause_generation_mode),
        rollout_engines=engines,
        weight_version=7,
    )


def _run(method, updater: SimpleNamespace, *, rank: int = 0) -> None:
    with patch(f"{_MIXIN_MODULE}.dist") as dist_mock:
        dist_mock.get_rank.return_value = rank
        method(updater)


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
    method,
    updater: SimpleNamespace,
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

    with patch(f"{_MIXIN_MODULE}.async_utils.submit", side_effect=observed_submit):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = executor.submit(_run, method, updater)
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


class TestPauseAndPrepareEngines:
    """Opening the update session must pause, flush and only then open the session, on every engine."""

    @pytest.mark.parametrize("pause_generation_mode", ["retract", "abort"])
    def test_every_engine_is_paused_flushed_and_moved_into_an_update_session(self, pause_generation_mode):
        """Loading weights into a still-generating or still-closed engine corrupts the rollout."""
        calls: list[tuple[int, str, dict]] = []
        phases = ["pause_generation", "flush_cache", "begin_weight_update"]
        gates = {phase: threading.Event() for phase in phases}
        updater = _make_updater(
            calls,
            later_engine_gates=gates,
            pause_generation_mode=pause_generation_mode,
        )

        _run_with_gated_later_engine(
            DistBucketedWeightUpdateMixin._pause_and_prepare_engines,
            updater,
            calls,
            phases,
            gates,
        )

        assert _phases(calls) == phases
        for method in phases:
            assert _engines_called(calls, method) == list(range(_ENGINE_COUNT))
        assert _kwargs_of(calls, "pause_generation") == [{"mode": pause_generation_mode}] * _ENGINE_COUNT
        assert _kwargs_of(calls, "begin_weight_update") == [{"selector": "all"}] * _ENGINE_COUNT
        assert updater._weight_update_selector == "all"

    def test_in_place_pause_mode_skips_the_flush(self):
        """in_place pause keeps the running requests, so their cache must survive."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, pause_generation_mode="in_place")

        _run(DistBucketedWeightUpdateMixin._pause_and_prepare_engines, updater)

        assert _phases(calls) == ["pause_generation", "begin_weight_update"]
        assert _kwargs_of(calls, "pause_generation") == [{"mode": "in_place"}] * _ENGINE_COUNT

    def test_a_failed_pause_opens_no_update_session(self):
        """A session opened on an engine that never paused would load weights under generation."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, failing_method="pause_generation")

        with pytest.raises(RuntimeError, match="pause_generation failed"):
            _run(DistBucketedWeightUpdateMixin._pause_and_prepare_engines, updater)

        assert _phases(calls) == ["pause_generation"]
        assert _engines_called(calls, "pause_generation") == [1]

    def test_a_failed_flush_opens_no_update_session(self):
        """A session opened over a stale cache would serve tokens generated from the old weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, failing_method="flush_cache")

        with pytest.raises(RuntimeError, match="flush_cache failed"):
            _run(DistBucketedWeightUpdateMixin._pause_and_prepare_engines, updater)

        assert _phases(calls) == ["pause_generation", "flush_cache"]
        assert _engines_called(calls, "flush_cache") == [1]

    def test_a_failed_begin_prevents_the_update_from_starting(self):
        """A begin failure must escape instead of letting weight transfer proceed with a closed engine."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, failing_method="begin_weight_update", failing_engine_index=1)

        with pytest.raises(RuntimeError, match="begin_weight_update failed"):
            _run(DistBucketedWeightUpdateMixin._pause_and_prepare_engines, updater)

        assert _phases(calls) == ["pause_generation", "flush_cache", "begin_weight_update"]
        assert _engines_called(calls, "begin_weight_update") == [0]

    def test_non_source_rank_issues_no_requests(self):
        """Every rank runs the updater, but only rank 0 may drive the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls)

        _run(DistBucketedWeightUpdateMixin._pause_and_prepare_engines, updater, rank=1)

        assert calls == []


class TestFinalizeAndResumeEngines:
    """Closing the update session must publish the version, close, and only then resume, on every engine."""

    def test_every_engine_gets_the_version_then_the_close_then_the_resume(self):
        """Resuming before the session closes would serve half-loaded weights."""
        calls: list[tuple[int, str, dict]] = []
        phases = ["update_weight_version", "end_weight_update", "continue_generation"]
        gates = {phase: threading.Event() for phase in phases}
        updater = _make_updater(calls, later_engine_gates=gates)

        _run_with_gated_later_engine(
            DistBucketedWeightUpdateMixin._finalize_and_resume_engines,
            updater,
            calls,
            phases,
            gates,
        )

        assert _phases(calls) == phases
        for method in phases:
            assert _engines_called(calls, method) == list(range(_ENGINE_COUNT))
        assert _kwargs_of(calls, "update_weight_version") == [{"weight_version": "7"}] * _ENGINE_COUNT

    def test_a_failed_version_publication_neither_closes_the_session_nor_resumes(self):
        """An engine resuming under a version it never acknowledged would mislabel its samples."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, failing_method="update_weight_version")

        with pytest.raises(RuntimeError, match="update_weight_version failed"):
            _run(DistBucketedWeightUpdateMixin._finalize_and_resume_engines, updater)

        assert _phases(calls) == ["update_weight_version"]
        assert _engines_called(calls, "update_weight_version") == [1]

    def test_a_failed_session_close_does_not_resume_generation(self):
        """An engine that resumed without a post-load pass would serve packed weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls, failing_method="end_weight_update")

        with pytest.raises(RuntimeError, match="end_weight_update failed"):
            _run(DistBucketedWeightUpdateMixin._finalize_and_resume_engines, updater)

        assert _phases(calls) == ["update_weight_version", "end_weight_update"]
        assert _engines_called(calls, "end_weight_update") == [1]

    def test_non_source_rank_issues_no_requests(self):
        """Every rank runs the updater, but only rank 0 may drive the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(calls)

        _run(DistBucketedWeightUpdateMixin._finalize_and_resume_engines, updater, rank=1)

        assert calls == []
