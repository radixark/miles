"""A failed step raises NonGlobalFatalError, the type the actor boundary converts to a result."""

import pytest

from miles.backends.training_utils.checkpoint_io import NonGlobalFatalError, run_with_failure_collective


def test_a_failed_step_raises_the_non_global_fatal_type():
    def step():
        raise ValueError("bad shard")

    with pytest.raises(NonGlobalFatalError, match="ValueError: bad shard"):
        run_with_failure_collective(step)
