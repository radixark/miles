import os
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _observation
from tests.utils.soak.core import event_log as event_log_module
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakCollectionClosedEvent, StoredEvent, read_events


class TestEventLogCreation:
    def test_an_existing_log_is_never_overwritten(self, tmp_path: Path) -> None:
        """A rerun reusing an evidence path must fail instead of truncating the earlier evidence."""
        path = tmp_path / "events.jsonl"
        path.write_text("earlier\n")

        with pytest.raises(FileExistsError):
            EventLog(path)
        assert path.read_text() == "earlier\n"

    def test_missing_parent_directories_are_created(self, tmp_path: Path) -> None:
        """A fresh evidence directory is created together with the empty log."""
        path = tmp_path / "a" / "b" / "events.jsonl"

        log = EventLog(path)

        assert path.read_text() == ""
        assert log.path == path
        assert log.events == []


class TestEventLogAppend:
    def test_appended_events_are_persisted_with_consecutive_sequences(self, tmp_path: Path) -> None:
        """Each append writes one stored line numbered by its position in the log."""
        path = tmp_path / "events.jsonl"
        log = EventLog(path)
        first = _observation(None, at=_at(0))
        second = _observation([_cell_target()], at=_at(1))

        log.append(first)
        log.append(second)

        stored = [StoredEvent.model_validate_json(line) for line in path.read_text().splitlines()]
        assert [one.sequence for one in stored] == [0, 1]
        assert [one.event for one in stored] == [first, second] == log.events
        assert read_events(path, require_closed=False) == log.events

    def test_a_later_mutation_of_the_appended_event_does_not_change_the_evidence(self, tmp_path: Path) -> None:
        """The log keeps a deep snapshot so mutable fields of the caller's event cannot rewrite history."""
        log = EventLog(tmp_path / "events.jsonl")
        observation = _observation(None, at=_at(0), errors={"observer": "slow"})

        log.append(observation)
        observation.errors["observer"] = "rewritten"

        assert log.events[0].errors == {"observer": "slow"}

    def test_the_events_property_returns_a_copy(self, tmp_path: Path) -> None:
        """Callers cannot append to the in-memory evidence without writing it to disk."""
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_observation(None, at=_at(0)))

        log.events.append(SoakCollectionClosedEvent(timestamp=_at(1)))

        assert len(log.events) == 1

    def test_every_append_is_fsynced_before_it_becomes_visible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An event only enters memory after its line was flushed and fsynced to disk."""
        log = EventLog(tmp_path / "events.jsonl")
        synced_lines: list[int] = []

        def fsync(fileno: int) -> None:
            synced_lines.append(len(log.path.read_text().splitlines()))
            assert len(log.events) == len(synced_lines) - 1

        monkeypatch.setattr(event_log_module.os, "fsync", fsync)

        log.append(_observation(None, at=_at(0)))
        log.append(_observation(None, at=_at(1)))

        assert synced_lines == [1, 2]

    def test_an_event_whose_fsync_failed_is_not_kept_in_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed durable write must not leave memory ahead of what the caller was told."""
        log = EventLog(tmp_path / "events.jsonl")

        def fsync(fileno: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(event_log_module.os, "fsync", fsync)

        with pytest.raises(OSError, match="disk full"):
            log.append(_observation(None, at=_at(0)))
        assert log.events == []

    def test_appending_after_closure_is_refused_and_leaves_the_file_unchanged(self, tmp_path: Path) -> None:
        """A closed collection accepts no further evidence."""
        log = EventLog(tmp_path / "events.jsonl")
        log.append(SoakCollectionClosedEvent(timestamp=_at(0)))
        before = log.path.read_bytes()

        with pytest.raises(AssertionError, match="closed"):
            log.append(_observation(None, at=_at(1)))
        assert log.path.read_bytes() == before
        assert len(log.events) == 1

    def test_appending_to_a_log_whose_file_grew_elsewhere_still_writes_at_the_end(self, tmp_path: Path) -> None:
        """Appends seek to the end so they never overwrite bytes already on disk."""
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_observation(None, at=_at(0)))
        with log.path.open("a") as stream:
            stream.write("foreign\n")

        log.append(_observation(None, at=_at(1)))

        lines = log.path.read_text().splitlines()
        assert lines[1] == "foreign"
        assert StoredEvent.model_validate_json(lines[2]).sequence == 1
        assert os.path.getsize(log.path) == sum(len(line) + 1 for line in lines)
