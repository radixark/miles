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
                data_rows=MIXED_DATA_ROWS,
            ),
            id="sample_filter_vs_all_samples",
        ),
    ],
    indirect=True,
)
def test_sample_filter_and_all_samples_process(rollout_integration_env):
    env = rollout_integration_env
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
