from pathlib import Path

import pytest

from miles.utils.simple_checkpointer import atomic_save_folder, load_simple_checkpoint, save_simple_checkpoint


def test_a_saved_state_round_trips_through_its_directory(tmp_path: Path) -> None:
    """The file a component writes is the file the same component reads back."""
    directory = tmp_path / "component"

    save_simple_checkpoint(directory=directory, data={"cursor": 7})

    assert load_simple_checkpoint(directory=directory) == {"cursor": 7}


def test_a_missing_file_inside_a_directory_is_refused(tmp_path: Path) -> None:
    """A published directory holds every file, so a missing one is corruption rather than a fresh start."""
    with pytest.raises(AssertionError, match="state.pt"):
        load_simple_checkpoint(directory=tmp_path)


def test_a_directory_appears_only_once_every_file_inside_it_is_written(tmp_path: Path) -> None:
    """A half-written checkpoint under the published name would be restored as if it were whole."""
    target = tmp_path / "rollout" / "2"

    with atomic_save_folder(target) as dir_temp:
        (dir_temp / "state.pt").write_text("state")
        assert not target.exists()

    assert [one.name for one in target.iterdir()] == ["state.pt"]


def test_an_interrupted_save_publishes_nothing_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """A crash mid-save must leave no directory a resume would trust."""
    target = tmp_path / "rollout" / "2"

    with pytest.raises(RuntimeError, match="save interrupted"), atomic_save_folder(target) as dir_temp:
        (dir_temp / "state.pt").write_text("state")
        raise RuntimeError("save interrupted")

    assert not target.exists()
    assert list(target.parent.glob(".tmp-*")) == []


def test_a_temporary_directory_of_a_dead_save_is_removed_by_the_next_one(tmp_path: Path) -> None:
    """A save killed before it could clean up must not leave rubbish growing in the checkpoint root."""
    target = tmp_path / "rollout" / "2"
    target.parent.mkdir(parents=True)
    (target.parent / ".tmp-2-999999").mkdir()

    with atomic_save_folder(target):
        pass

    assert list(target.parent.glob(".tmp-*")) == []


def test_saving_the_same_target_again_replaces_the_published_directory(tmp_path: Path) -> None:
    """A re-save has to land whole, so the old directory is swapped out rather than written into."""
    target = tmp_path / "rollout" / "2"
    with atomic_save_folder(target) as dir_temp:
        (dir_temp / "first.pt").write_text("first")

    with atomic_save_folder(target) as dir_temp:
        (dir_temp / "second.pt").write_text("second")

    assert [one.name for one in target.iterdir()] == ["second.pt"]
