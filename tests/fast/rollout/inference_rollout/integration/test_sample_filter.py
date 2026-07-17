from unittest.mock import Mock

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.misc import function_registry

# Data with only 2 reward=1 samples out of 4.
# This ensures all 4 samples must be generated to collect 2 valid ones.
_FILTER_TEST_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},  # reward=1
    {"input": "What is 1+8?", "label": "wrong"},  # reward=0
    {"input": "What is 1+9?", "label": "wrong"},  # reward=0
    {"input": "What is 1+6?", "label": "7"},  # reward=1
]


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(
                [
                    "--rollout-batch-size",
                    "2",
                    "--over-sampling-batch-size",
                    "4",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                    "--rollout-sample-filter-path",
                    "test:sample_filter",
                    "--rollout-all-samples-process-path",
                    "test:all_samples_process",
                ],
                data_rows=_FILTER_TEST_DATA_ROWS,
            ),
            id="sample_filter_vs_all_samples",
        ),
    ],
    indirect=True,
)
def test_sample_filter_and_all_samples_process(rollout_env):
    env = rollout_env
    sample_filter_mock = Mock()
    all_samples_process_mock = Mock()

    with (
        function_registry.temporary("test:filter_by_reward", filter_by_reward),
        function_registry.temporary("test:sample_filter", sample_filter_mock),
        function_registry.temporary("test:all_samples_process", all_samples_process_mock),
    ):
        load_and_call_train(env.args, env.data_source)

    sample_filter_mock.assert_called_once()
    _, filtered_data = sample_filter_mock.call_args[0]
    rewards = [g[0][0].reward if isinstance(g[0], list) else g[0].reward for g in filtered_data]
    assert all(r == 1 for r in rewards)

    all_samples_process_mock.assert_called_once()
    _, all_samples, data_source = all_samples_process_mock.call_args[0]
    assert data_source is not None

    assert len(all_samples) > len(filtered_data), "all_samples_process should see more samples than sample_filter"
