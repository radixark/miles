from unittest.mock import Mock

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
                    "--rollout-batch-size",
                    "3",
                    "--over-sampling-batch-size",
                    "6",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                    "--rollout-all-samples-process-path",
                    "test:all_samples_process",
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
    all_samples_process_mock = Mock()

    with (
        function_registry.temporary("test:filter_by_reward", filter_by_reward),
        function_registry.temporary("test:all_samples_process", all_samples_process_mock),
    ):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)

    _, all_samples, _ = all_samples_process_mock.call_args[0]
    assert len(all_samples) > len(out.samples), "Over sampling should generate more samples than output"
    all_rewards = {g[0].reward for g in all_samples}
    assert 0 in all_rewards, "Some samples should have been filtered out"
