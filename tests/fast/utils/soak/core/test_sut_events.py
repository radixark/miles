import os
from pathlib import Path

from tests.fast.utils.soak.soak_fakes import (
    _at,
    _cell_target,
    _observation,
    _reconfigure,
    _step_end,
    _sut_main_source,
    _write_sut_lines,
)
from tests.utils.soak.core.sut_events import SutEventFeed, SutFileKey

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, MetricEvent, TrainGroupStepEndEvent


def _feed(directory: Path) -> SutEventFeed:
    return SutEventFeed(
        directory=directory,
        file_patterns=("trainer_controller_*.jsonl", "rollout_executor.jsonl"),
        event_types=(TrainGroupStepEndEvent, CellReconfigureEvent),
    )


class TestSutEventFeedAttach:
    async def test_consecutive_attaches_deliver_each_complete_line_exactly_once(self, tmp_path: Path) -> None:
        """A second attach returns only lines written after the first one."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        _write_sut_lines(path, [_step_end(0, at=_at(0))])
        feed = _feed(tmp_path)

        first = await feed.attach(_observation(None, at=_at(1)))
        unchanged = await feed.attach(_observation(None, at=_at(2)))
        _write_sut_lines(path, [_step_end(1, at=_at(3))])
        later = await feed.attach(_observation(None, at=_at(4)))

        assert [event.rollout_id for event in first.new_sut_events] == [0]
        assert unchanged.new_sut_events == []
        assert [event.rollout_id for event in later.new_sut_events] == [1]

    async def test_a_partial_last_line_waits_until_it_is_completed(self, tmp_path: Path) -> None:
        """A line still being written is neither parsed nor counted as consumed."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        pending = _step_end(1, at=_at(1)).model_dump_json()
        _write_sut_lines(path, [_step_end(0, at=_at(0))], trailing=pending[:10])
        feed = _feed(tmp_path)

        first = await feed.attach(_observation(None, at=_at(2)))
        with path.open("a") as stream:
            stream.write(pending[10:] + "\n")
        second = await feed.attach(_observation(None, at=_at(3)))

        assert [event.rollout_id for event in first.new_sut_events] == [0]
        assert [event.rollout_id for event in second.new_sut_events] == [1]

    async def test_a_file_replaced_under_the_same_name_is_read_from_its_start(self, tmp_path: Path) -> None:
        """A new inode is a new file, so its lines are not skipped by the old file's cursor."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        _write_sut_lines(path, [_step_end(0, at=_at(0)), _step_end(1, at=_at(1))])
        feed = _feed(tmp_path)
        await feed.attach(_observation(None, at=_at(2)))

        replacement = tmp_path / "replacement.tmp"
        _write_sut_lines(replacement, [_step_end(5, at=_at(3)), _step_end(6, at=_at(4)), _step_end(7, at=_at(5))])
        os.replace(replacement, path)
        after = await feed.attach(_observation(None, at=_at(6)))

        assert [event.rollout_id for event in after.new_sut_events] == [5, 6, 7]

    async def test_a_file_truncated_in_place_is_read_from_its_start(self, tmp_path: Path) -> None:
        """A file shorter than the cursor was rewritten, so none of its lines are skipped."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        _write_sut_lines(path, [_step_end(0, at=_at(0)), _step_end(1, at=_at(1))])
        feed = _feed(tmp_path)
        await feed.attach(_observation(None, at=_at(2)))

        with path.open("w") as stream:
            stream.write(_step_end(9, at=_at(3)).model_dump_json() + "\n")
        after = await feed.attach(_observation(None, at=_at(4)))

        assert [event.rollout_id for event in after.new_sut_events] == [9]

    async def test_only_configured_files_and_event_types_are_delivered(self, tmp_path: Path) -> None:
        """Files outside the patterns and events outside the types never reach the soak."""
        _write_sut_lines(
            tmp_path / "trainer_controller_actor.jsonl",
            [MetricEvent(timestamp=_at(0), source=_sut_main_source(), metrics={}), _step_end(0, at=_at(1))],
        )
        _write_sut_lines(
            tmp_path / "rollout_executor.jsonl",
            [_reconfigure(at=_at(2), healed_cell_indices=[1], cell_incarnations_after={})],
        )
        _write_sut_lines(tmp_path / "train_actor_0.jsonl", [_step_end(7, at=_at(3))])

        observed = await _feed(tmp_path).attach(_observation(None, at=_at(4)))

        assert [type(event) for event in observed.new_sut_events] == [CellReconfigureEvent, TrainGroupStepEndEvent]

    async def test_the_observation_keeps_its_targets_and_gains_no_errors_on_success(self, tmp_path: Path) -> None:
        """Attaching events must not alter what the observer saw."""
        _write_sut_lines(tmp_path / "trainer_controller_actor.jsonl", [_step_end(0, at=_at(0))])
        observation = _observation([_cell_target()], at=_at(1), errors={"pods": "slow"})

        observed = await _feed(tmp_path).attach(observation)

        assert observed.targets == observation.targets
        assert observed.timestamp == observation.timestamp
        assert observed.errors == {"pods": "slow"}

    async def test_the_cursor_is_keyed_by_file_name_and_inode(self, tmp_path: Path) -> None:
        """The consumed line count is remembered per file identity."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        _write_sut_lines(path, [_step_end(0, at=_at(0)), _step_end(1, at=_at(1))])
        feed = _feed(tmp_path)

        await feed.attach(_observation(None, at=_at(2)))

        assert feed.cursor == {SutFileKey(name=path.name, inode=path.stat().st_ino): 2}


class TestSutEventFeedFailures:
    async def test_a_missing_directory_is_recorded_as_an_error(self, tmp_path: Path) -> None:
        """An absent events directory is an observation error, not an empty event stream."""
        feed = _feed(tmp_path / "missing")

        observed = await feed.attach(_observation(None, at=_at(0), errors={"pods": "slow"}))

        assert set(observed.errors) == {"pods", "sut_events"}
        assert observed.new_sut_events == []
        assert feed.cursor == {}

    async def test_an_empty_directory_delivers_no_events_without_error(self, tmp_path: Path) -> None:
        """Before training writes anything the feed is empty but healthy."""
        observed = await _feed(tmp_path).attach(_observation(None, at=_at(0)))

        assert observed.new_sut_events == []
        assert observed.errors == {}

    async def test_an_unparsable_line_is_an_error_and_does_not_advance_the_cursor(self, tmp_path: Path) -> None:
        """Lines around a corrupt one are delivered only once it is fixed, never silently skipped."""
        path = tmp_path / "trainer_controller_actor.jsonl"
        _write_sut_lines(path, [_step_end(0, at=_at(0))], trailing="{not json\n")
        feed = _feed(tmp_path)

        failed = await feed.attach(_observation(None, at=_at(1)))
        path.write_text(
            _step_end(0, at=_at(0)).model_dump_json() + "\n" + _step_end(1, at=_at(1)).model_dump_json() + "\n"
        )
        fixed = await feed.attach(_observation(None, at=_at(2)))

        assert "sut_events" in failed.errors
        assert failed.new_sut_events == []
        assert [event.rollout_id for event in fixed.new_sut_events] == [0, 1]

    async def test_an_unknown_event_type_is_an_error(self, tmp_path: Path) -> None:
        """A line that is JSON but no known event fails validation instead of being dropped."""
        (tmp_path / "trainer_controller_actor.jsonl").write_text('{"type": "not_an_event"}\n')

        observed = await _feed(tmp_path).attach(_observation(None, at=_at(0)))

        assert "sut_events" in observed.errors
