import pytest
from tests.rollout.modular_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    config,
    filter_by_reward,
    load_and_call_train,
)

from miles.utils.misc import function_registry


@pytest.mark.parametrize(
    "rollout_integration_env",
    [
        pytest.param(
            config(
                [
                    "--over-sampling-batch-size",
                    "2",
                    "--rollout-batch-size",
                    "3",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                ],
                data_rows=MIXED_DATA_ROWS,
            ),
            id="over_sampling_with_filter",
        ),
    ],
    indirect=["rollout_integration_env"],
)
def test_over_sampling_collects_enough_samples(rollout_integration_env):
    env = rollout_integration_env
    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)
