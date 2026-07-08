"""Tests for miles.utils.audit_utils.event_logger.checkpoint."""

from argparse import Namespace
from pathlib import Path

from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint


def _args(*, event_dir: Path | None, save: Path | None = None, load: Path | None = None) -> Namespace:
    return Namespace(
        save_debug_event_data=str(event_dir) if event_dir else None,
        save=str(save) if save else None,
        load=str(load) if load else None,
    )


def _write_tracker(ckpt: Path, content: str) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "latest_checkpointed_iteration.txt").write_text(content)


class TestSnapshotRestoreRoundtrip:
    def test_restore_replaces_live_dir_with_snapshot(self, tmp_path: Path) -> None:
        """A resumed run sees exactly the snapshotted events, not the live dir's leftovers."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("committed\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=3)

        # Events written after the save (would be re-executed by the resumed run).
        (events / "main.jsonl").write_text("committed\nrewound-future\n")
        (events / "straggler.jsonl").write_text("late\n")
        _write_tracker(ckpt, "3")
        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "committed\n"
        assert not (events / "straggler.jsonl").exists()

    def test_snapshot_overwrites_previous_snapshot_of_same_iteration(self, tmp_path: Path) -> None:
        """Re-saving the same iteration replaces its snapshot."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("v1\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)
        (events / "main.jsonl").write_text("v2\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)

        assert (ckpt / "iter_0000001" / "debug_events" / "main.jsonl").read_text() == "v2\n"


class TestNoOpCases:
    def test_restore_skips_when_not_resuming(self, tmp_path: Path) -> None:
        """No --load means no restore."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_when_checkpoint_has_no_snapshot(self, tmp_path: Path) -> None:
        """Checkpoints predating event snapshots leave the live dir untouched."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "2")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_release_tracker(self, tmp_path: Path) -> None:
        """A non-numeric tracker (e.g. 'release') is not a resumable iteration."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "release")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_snapshot_skips_when_events_disabled_or_no_save(self, tmp_path: Path) -> None:
        """Disabled events or no save dir means no snapshot."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.snapshot(_args(event_dir=None, save=tmp_path / "ckpt"), iteration=1)
        event_logger_checkpoint.snapshot(_args(event_dir=events), iteration=1)

        assert not (tmp_path / "ckpt").exists()
