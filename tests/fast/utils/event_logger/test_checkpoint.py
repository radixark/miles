"""Tests for miles.utils.audit_utils.event_logger.checkpoint."""

from argparse import Namespace
from pathlib import Path

import pytest

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
    def test_restore_without_a_checkpoint_discards_events_from_the_abandoned_attempt(self, tmp_path: Path) -> None:
        """Restarting from the initial state discards audit events from the abandoned attempt."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("abandoned-rollout-0\nabandoned-rollout-1\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert not events.exists()
        assert list(tmp_path.glob(".trash_*")) == []

    def test_restore_without_a_checkpoint_accepts_an_absent_event_dir(self, tmp_path: Path) -> None:
        """A first launch has no abandoned event directory to discard."""
        events = tmp_path / "events"

        event_logger_checkpoint.restore(_args(event_dir=events, load=tmp_path / "ckpt"))

        assert not events.exists()

    def test_restore_without_a_checkpoint_discards_only_the_configured_event_dir(self, tmp_path: Path) -> None:
        """Resetting audit state leaves sibling run evidence untouched."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("abandoned\n")
        sibling = tmp_path / "sibling.jsonl"
        sibling.write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=tmp_path / "ckpt"))

        assert not events.exists()
        assert sibling.read_text() == "keep\n"

    def test_restore_without_a_checkpoint_propagates_discard_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed reset stops relaunch instead of accepting stale audit events."""
        events = tmp_path / "events"
        events.mkdir()

        def fail_discard(path: Path) -> None:
            assert path == events
            raise OSError("discard failed")

        monkeypatch.setattr(event_logger_checkpoint.shutil, "rmtree", fail_discard)

        with pytest.raises(OSError, match="discard failed"):
            event_logger_checkpoint.restore(_args(event_dir=events, load=tmp_path / "ckpt"))

        assert events.exists()

    @pytest.mark.parametrize("target_exists", [False, True], ids=["dangling", "live"])
    def test_restore_without_a_checkpoint_rejects_a_symlink_event_dir(
        self, tmp_path: Path, target_exists: bool
    ) -> None:
        """Resetting audit state refuses both live and dangling event-directory symlinks."""
        target = tmp_path / "target"
        if target_exists:
            target.mkdir()
            (target / "main.jsonl").write_text("keep\n")
        events = tmp_path / "events"
        events.symlink_to(target, target_is_directory=True)

        with pytest.raises(RuntimeError, match="symbolic link"):
            event_logger_checkpoint.restore(_args(event_dir=events, load=tmp_path / "ckpt"))

        assert events.is_symlink()
        assert target.exists() is target_exists
        if target_exists:
            assert (target / "main.jsonl").read_text() == "keep\n"

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

    def test_restore_from_release_tracker_discards_live_events(self, tmp_path: Path) -> None:
        """A release checkpoint has no event snapshot, so relaunch returns to initial audit state."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "release")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert not events.exists()

    def test_snapshot_skips_when_events_disabled_or_no_save(self, tmp_path: Path) -> None:
        """Disabled events or no save dir means no snapshot."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.snapshot(_args(event_dir=None, save=tmp_path / "ckpt"), iteration=1)
        event_logger_checkpoint.snapshot(_args(event_dir=events), iteration=1)

        assert not (tmp_path / "ckpt").exists()
