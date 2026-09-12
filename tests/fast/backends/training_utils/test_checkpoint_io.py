"""Only local filesystem failures become coordinated checkpoint errors."""

import pytest

from miles.backends.training_utils.checkpoint_io import CheckpointIOError, run_local_io_collective


@pytest.mark.parametrize("error, expected", [(OSError("disk full"), CheckpointIOError), (RuntimeError("collective failed"), RuntimeError)])
def test_only_local_io_errors_are_converted(error, expected):
    def step():
        raise error

    with pytest.raises(expected, match=str(error)) as caught:
        run_local_io_collective(step)
    if expected is RuntimeError:
        assert caught.value is error
