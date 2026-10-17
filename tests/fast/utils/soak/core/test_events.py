from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.fast.utils.soak.soak_fakes import (
    _applied,
    _at,
    _cell_target,
    _observation,
    _request,
    _requested,
    _result,
    _step_end,
    _stored_line,
)
from tests.utils.soak.core.events import (
    SoakCollectionClosedEvent,
    SoakEvent,
    SoakEvidenceArchivedEvent,
    StoredEvent,
    file_sha256,
    read_events,
)


class TestSoakEventSchema:
    def test_every_action_event_round_trips_through_the_discriminated_union(self) -> None:
        """Requested, applied, result and observation events are restored as their own types."""
        request = _request(_cell_target())
        events: list[SoakEvent] = [
            _requested(request, at=_at(0)),
            _applied(request, at=_at(1)),
            _result(request, at=_at(2), returned=False, error="boom"),
            _observation([_cell_target()], at=_at(3), new_sut_events=[_step_end(1, at=_at(3))]),
        ]
        adapter = TypeAdapter(list[SoakEvent])

        assert adapter.validate_json(adapter.dump_json(events)) == events

    def test_an_unknown_event_kind_is_rejected(self) -> None:
        """Evidence written by another schema must not be silently coerced into a soak event."""
        with pytest.raises(ValidationError):
            TypeAdapter(SoakEvent).validate_python({"kind": "observed", "targets": None})

    def test_an_unknown_event_field_is_rejected(self) -> None:
        """Strict events refuse extra fields so a renamed field cannot be dropped on replay."""
        payload = SoakCollectionClosedEvent(timestamp=_at(0)).model_dump(mode="json")

        with pytest.raises(ValidationError):
            TypeAdapter(SoakEvent).validate_python({**payload, "reason": "late"})

    def test_a_stored_event_of_another_version_is_rejected(self) -> None:
        """A version bump must fail replay instead of reading an incompatible layout."""
        payload = StoredEvent(sequence=0, event=SoakCollectionClosedEvent(timestamp=_at(0))).model_dump(mode="json")

        with pytest.raises(ValidationError):
            StoredEvent.model_validate({**payload, "version": 2})


class TestReadEvents:
    def test_a_closed_log_replays_every_event_in_order(self, tmp_path: Path) -> None:
        """A complete log returns its events in sequence order ending in the closing event."""
        path = tmp_path / "events.jsonl"
        events: list[SoakEvent] = [_observation(None, at=_at(0)), SoakCollectionClosedEvent(timestamp=_at(1))]
        path.write_text("".join(_stored_line(index, event) for index, event in enumerate(events)))

        assert read_events(path) == events

    def test_a_gap_in_the_sequence_is_rejected(self, tmp_path: Path) -> None:
        """A dropped line leaves a sequence gap that replay must refuse."""
        path = tmp_path / "events.jsonl"
        path.write_text(
            _stored_line(0, _observation(None, at=_at(0)))
            + _stored_line(2, SoakCollectionClosedEvent(timestamp=_at(1)))
        )

        with pytest.raises(AssertionError, match="Missing or reordered"):
            read_events(path)

    def test_reordered_lines_are_rejected(self, tmp_path: Path) -> None:
        """Swapped lines are detected through their stored sequence numbers."""
        path = tmp_path / "events.jsonl"
        path.write_text(
            _stored_line(1, SoakCollectionClosedEvent(timestamp=_at(1)))
            + _stored_line(0, _observation(None, at=_at(0)))
        )

        with pytest.raises(AssertionError, match="Missing or reordered"):
            read_events(path)

    def test_an_event_after_closure_is_rejected(self, tmp_path: Path) -> None:
        """Nothing may be appended to evidence once collection has closed."""
        path = tmp_path / "events.jsonl"
        path.write_text(
            _stored_line(0, SoakCollectionClosedEvent(timestamp=_at(0)))
            + _stored_line(1, _observation(None, at=_at(1)))
        )

        with pytest.raises(AssertionError, match="after closure"):
            read_events(path, require_closed=False)

    def test_an_unclosed_log_is_incomplete_evidence(self, tmp_path: Path) -> None:
        """A log whose writer died before closing is refused as incomplete."""
        path = tmp_path / "events.jsonl"
        path.write_text(_stored_line(0, _observation(None, at=_at(0))))

        with pytest.raises(AssertionError, match="incomplete"):
            read_events(path)

    def test_an_unclosed_log_is_readable_when_closure_is_not_required(self, tmp_path: Path) -> None:
        """The collector reads its own open log back before closing it."""
        path = tmp_path / "events.jsonl"
        observation = _observation(None, at=_at(0))
        path.write_text(_stored_line(0, observation))

        assert read_events(path, require_closed=False) == [observation]

    def test_an_empty_log_is_incomplete_evidence(self, tmp_path: Path) -> None:
        """A created but never written log is not a closed soak."""
        path = tmp_path / "events.jsonl"
        path.touch()

        with pytest.raises(AssertionError, match="incomplete"):
            read_events(path)

    def test_a_truncated_last_line_is_rejected(self, tmp_path: Path) -> None:
        """A torn final write must fail replay instead of dropping the partial event."""
        path = tmp_path / "events.jsonl"
        closed = _stored_line(1, SoakCollectionClosedEvent(timestamp=_at(1)))
        path.write_text(_stored_line(0, _observation(None, at=_at(0))) + closed[: len(closed) // 2])

        with pytest.raises(ValidationError):
            read_events(path)


class TestArchivedFileIntegrity:
    def _write_archived_log(self, tmp_path: Path, *, relative: str, expected_sha256: str) -> Path:
        path = tmp_path / "events.jsonl"
        archived = SoakEvidenceArchivedEvent(
            timestamp=_at(0),
            sources={"training_events": tmp_path / "sources"},
            missing_sources=[],
            sha256_of_file={relative: expected_sha256},
        )
        path.write_text(_stored_line(0, archived) + _stored_line(1, SoakCollectionClosedEvent(timestamp=_at(1))))
        return path

    def test_an_unchanged_archived_file_is_accepted(self, tmp_path: Path) -> None:
        """Replay succeeds when every archived file still has its recorded hash."""
        archived = tmp_path / "sources" / "a.jsonl"
        archived.parent.mkdir()
        archived.write_text("line\n")
        path = self._write_archived_log(tmp_path, relative="sources/a.jsonl", expected_sha256=file_sha256(archived))

        assert len(read_events(path)) == 2

    def test_a_modified_archived_file_is_rejected(self, tmp_path: Path) -> None:
        """Editing archived evidence after the run is detected through its hash."""
        archived = tmp_path / "sources" / "a.jsonl"
        archived.parent.mkdir()
        archived.write_text("line\n")
        path = self._write_archived_log(tmp_path, relative="sources/a.jsonl", expected_sha256=file_sha256(archived))
        archived.write_text("forged\n")

        with pytest.raises(AssertionError, match="changed"):
            read_events(path)

    def test_a_deleted_archived_file_is_rejected(self, tmp_path: Path) -> None:
        """Removing an archived file cannot pass as unchanged evidence."""
        archived = tmp_path / "sources" / "a.jsonl"
        archived.parent.mkdir()
        archived.write_text("line\n")
        path = self._write_archived_log(tmp_path, relative="sources/a.jsonl", expected_sha256=file_sha256(archived))
        archived.unlink()

        with pytest.raises(FileNotFoundError):
            read_events(path)

    def test_an_archived_path_escaping_the_log_directory_is_rejected(self, tmp_path: Path) -> None:
        """A recorded path cannot point the integrity check outside the archive."""
        outside = tmp_path / "outside.txt"
        outside.write_text("line\n")
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        path = self._write_archived_log(log_dir, relative="../outside.txt", expected_sha256=file_sha256(outside))

        with pytest.raises(AssertionError, match="escapes"):
            read_events(path)


class TestFileSha256:
    def test_the_digest_covers_content_larger_than_one_read_chunk(self, tmp_path: Path) -> None:
        """A change past the first megabyte still changes the recorded digest."""
        path = tmp_path / "big.bin"
        payload = bytearray(3 * 1024 * 1024)
        path.write_bytes(bytes(payload))
        before = file_sha256(path)
        payload[-1] = 1
        path.write_bytes(bytes(payload))

        assert file_sha256(path) != before
