from pathlib import Path

import pytest

from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint


def test_a_saved_state_round_trips_through_its_directory(tmp_path: Path) -> None:
    """The file a component writes is the file the same component reads back."""
    directory = tmp_path / "component"

    save_simple_checkpoint(directory=directory, data={"cursor": 7})

    assert load_simple_checkpoint(directory=directory) == {"cursor": 7}


def test_a_missing_file_inside_a_directory_is_refused(tmp_path: Path) -> None:
    """A published directory holds every file, so a missing one is corruption rather than a fresh start."""
    with pytest.raises(AssertionError, match="state.pt"):
        load_simple_checkpoint(directory=tmp_path)
