from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.simple_checkpointer import SimpleCheckpointer


@pytest.mark.parametrize("require_exists", [False, True])
def test_missing_checkpoint_respects_constructor_policy(tmp_path: Path, require_exists: bool) -> None:
    """Strict instances refuse missing state while optional instances retain fresh-start behavior."""
    checkpointer = SimpleCheckpointer(path_template="state_{rollout_id}.pt", require_exists=require_exists)
    args = Namespace(load=tmp_path, save=tmp_path)

    if require_exists:
        with pytest.raises(FileNotFoundError, match="state_3.pt"):
            checkpointer.load(args=args, rollout_id=3)
    else:
        assert checkpointer.load(args=args, rollout_id=3) is None

    checkpointer.save(args=args, rollout_id=3, data={"cursor": 7})
    assert checkpointer.load(args=args, rollout_id=3) == {"cursor": 7}


def test_strict_checkpoint_without_load_is_not_a_resume() -> None:
    """An absent load root disables restoration even for a mandatory state component."""
    checkpointer = SimpleCheckpointer(path_template="state_{rollout_id}.pt", require_exists=True)

    assert checkpointer.load(args=Namespace(load=None), rollout_id=None) is None
