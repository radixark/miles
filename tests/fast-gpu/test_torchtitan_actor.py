from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan; est_time is an unmeasured upper bound
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["torchtitan"], hardware=["hopper"], num_gpus=2)

from types import SimpleNamespace

import pytest

from miles.backends.torchtitan_utils.actor import _steps_per_rollout


@pytest.mark.parametrize("dynamic, steps", [(False, 4), (True, 1)])
def test_a_dynamic_global_batch_is_one_step_per_rollout(dynamic, steps):
    """The LR schedule length and the resumed start rollout count optimizer steps per rollout."""
    args = SimpleNamespace(
        rollout_batch_size=8, n_samples_per_prompt=4, global_batch_size=8, use_dynamic_global_batch_size=dynamic
    )
    assert _steps_per_rollout(args) == steps
