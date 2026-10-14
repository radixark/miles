import threading
from argparse import Namespace
from typing import Any

import pytest

from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    maybe_pause_engines,
    maybe_resume_engines,
    set_weight_version,
)


def _make_args(
    *, fully_async: bool = False, colocate: bool = False, pause_generation_mode: str = "abort"
) -> Namespace:
    return Namespace(
        fully_async=fully_async,
        colocate=colocate,
        pause_generation_mode=pause_generation_mode,
    )


class _RecordingClient:
    def __init__(self, *, result: Any = None, gate: threading.Event | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result if result is not None else {"success": True}
        self._gate = gate

    def __getattr__(self, name: str):
        async def method(**kwargs: Any) -> Any:
            if self._gate is not None:
                assert self._gate.wait(timeout=30)
            self.calls.append((name, kwargs))
            return self._result

        return method

    @property
    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class _FailingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        async def method(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            raise RuntimeError("boom")

        return method

    @property
    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _make_updaters(clients: list[Any]) -> list[Any]:
    cell_ids = [f"cell-{index}" for index in range(len(clients))]
    return list(create_rollout_cell_updaters(clients, cell_ids).values())


class TestWeightUpdateSessionFanOut:
    def test_begin_and_end_reach_every_cell(self) -> None:
        """A missed engine would load weights outside a session and corrupt them."""
        clients = [_RecordingClient() for _ in range(4)]
        updaters = _make_updaters(clients)

        begin_weight_update(updaters)
        end_weight_update(updaters)

        assert all(client.call_names == ["begin_weight_update", "end_weight_update"] for client in clients)

    def test_the_selector_and_the_base_sync_decision_reach_every_cell(self) -> None:
        """An adapter-only session on one engine and a base session on another would diverge the fleet."""
        clients = [_RecordingClient() for _ in range(2)]

        begin_weight_update(_make_updaters(clients), "draft", sync_base=False)

        for client in clients:
            assert client.calls == [("begin_weight_update", {"selector": "draft", "sync_base": False})]

    def test_the_lora_checksum_manifest_reaches_every_cell(self) -> None:
        """An unverified engine could serve an adapter that never finished streaming."""
        clients = [_RecordingClient(), _RecordingClient()]
        checksums = {"adapter": {"weight": "abc"}}

        end_weight_update(_make_updaters(clients), expected_lora_checksums=checksums)

        for client in clients:
            assert client.calls == [("end_weight_update", {"expected_lora_checksums": checksums})]


class TestSetWeightVersion:
    def test_every_cell_learns_the_new_version(self) -> None:
        """An engine left on the old version would be reported as serving stale weights."""
        clients = [_RecordingClient(), _RecordingClient()]

        set_weight_version(_make_updaters(clients), 11)

        for client in clients:
            assert client.calls == [("update_weight_version", {"weight_version": "11"})]


class TestPauseAndResume:
    @pytest.mark.parametrize("mode", ["abort", "stop"])
    def test_a_pause_that_discards_the_cache_flushes_every_cell(self, mode: str) -> None:
        """Weights written under a stale kv cache would be served with the old activations."""
        args = _make_args(pause_generation_mode=mode)
        clients = [_RecordingClient(), _RecordingClient()]

        maybe_pause_engines(args, _make_updaters(clients))

        for client in clients:
            assert client.call_names == ["pause_generation", "flush_cache"]
            assert client.calls[0][1] == {"mode": mode}

    def test_an_in_place_pause_keeps_the_cache_it_exists_to_preserve(self) -> None:
        """Flushing would discard exactly the kv cache that in_place pausing resumes against."""
        args = _make_args(pause_generation_mode="in_place")
        clients = [_RecordingClient(), _RecordingClient()]

        maybe_pause_engines(args, _make_updaters(clients))

        for client in clients:
            assert client.call_names == ["pause_generation"]

    def test_the_engines_are_left_alone_when_the_driver_owns_the_pause(self) -> None:
        """Pausing twice from two owners would resume generation while the weights are still moving."""
        args = _make_args(fully_async=True, colocate=True)
        clients = [_RecordingClient()]
        updaters = _make_updaters(clients)

        maybe_pause_engines(args, updaters)
        maybe_resume_engines(args, updaters)

        assert clients[0].calls == []

    def test_resume_reaches_every_cell(self) -> None:
        """An engine left paused would stop serving rollouts for the rest of the run."""
        args = _make_args()
        clients = [_RecordingClient(), _RecordingClient(), _RecordingClient()]

        maybe_resume_engines(args, _make_updaters(clients))

        for client in clients:
            assert client.call_names == ["continue_generation"]


class TestSessionFanOutGivesUpOnABrokenCell:
    """One cell losing the update must not take the session of every other cell with it."""

    def test_a_failing_cell_is_given_up_while_the_others_get_the_session(self) -> None:
        """One broken engine must not cost the healthy engines their weight update."""
        healthy = _RecordingClient()
        clients = [healthy, _FailingClient()]
        updaters = _make_updaters(clients)

        begin_weight_update(updaters)

        assert healthy.call_names == ["begin_weight_update"]
        assert not updaters[0].is_errored
        assert updaters[1].is_errored

    def test_every_cell_is_asked_before_a_failure_is_recorded(self) -> None:
        """The fan-out is concurrent, so a later engine must already be in flight when an earlier one fails."""
        gate = threading.Event()
        gate.set()
        slow = _RecordingClient(gate=gate)
        updaters = _make_updaters([_FailingClient(), slow])

        begin_weight_update(updaters)

        assert slow.call_names == ["begin_weight_update"]
        assert updaters[0].is_errored

    def test_a_cell_given_up_earlier_is_skipped_by_every_later_session_call(self) -> None:
        """An engine that lost the update must not be told the new version or put back into service."""
        clients = [_RecordingClient(), _RecordingClient()]
        updaters = _make_updaters(clients)
        updaters[1].mark_errored(RuntimeError("lost"))

        begin_weight_update(updaters)
        end_weight_update(updaters)
        set_weight_version(updaters, 7)
        maybe_resume_engines(_make_args(), updaters)

        assert clients[0].call_names == [
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
            "continue_generation",
        ]
        assert clients[1].calls == []


class TestEndWeightUpdate:

    def test_a_rejected_close_marks_that_cell_instead_of_raising_for_the_fleet(self) -> None:
        """Raising would abandon the engines that closed their session cleanly."""
        clients = [_RecordingClient(), _RecordingClient(result={"success": False, "message": "checksum mismatch"})]
        updaters = _make_updaters(clients)

        end_weight_update(updaters)

        assert not updaters[0].is_errored
        assert updaters[1].is_errored
        assert "checksum mismatch" in str(updaters[1]._error)

    def test_a_clean_close_leaves_every_cell_updatable(self) -> None:
        """A cell wrongly given up here would be dropped out of service for the rest of the run."""
        updaters = _make_updaters([_RecordingClient(), _RecordingClient()])

        end_weight_update(updaters)

        assert not any(updater.is_errored for updater in updaters)

    def test_a_cell_lost_during_the_pause_is_not_resumed(self) -> None:
        """Resuming an engine whose weights were never written would serve a half-written model."""
        args = _make_args()
        healthy = _RecordingClient()
        updaters = _make_updaters([healthy, _FailingClient()], args)

        maybe_pause_engines(args, updaters)
        maybe_resume_engines(args, updaters)

        assert healthy.call_names == ["pause_generation", "flush_cache", "continue_generation"]
        assert updaters[1].is_errored
