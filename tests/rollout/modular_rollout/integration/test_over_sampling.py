from unittest.mock import Mock

import pytest
from tests.rollout.modular_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    config,
    filter_by_reward,
    load_and_call_train,
)

from miles.utils.misc import function_registry

_BASE_ARGV = [
    "--over-sampling-batch-size",
    "6",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
    "--rollout-all-samples-process-path",
    "test:all_samples_process",
]


def _over_sampling_config(rollout_batch_size: int):
    return config(["--rollout-batch-size", str(rollout_batch_size)] + _BASE_ARGV, data_rows=MIXED_DATA_ROWS)


@pytest.mark.parametrize(
    "rollout_integration_env,min_expected_rounds",
    [
        pytest.param(_over_sampling_config(2), 1, id="one_round"),
        pytest.param(_over_sampling_config(3), 2, id="two_rounds"),
    ],
    indirect=["rollout_integration_env"],
)
def test_over_sampling_rounds(rollout_integration_env, min_expected_rounds):
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
    min_expected_all_samples = min_expected_rounds * env.args.over_sampling_batch_size
    assert len(all_samples) >= min_expected_all_samples, f"Expected at least {min_expected_rounds} round(s) of sampling"
    assert len(all_samples) > len(out.samples), "Over sampling should generate more samples than output"
    all_rewards = {g[0].reward for g in all_samples}
    assert 0 in all_rewards, "Some samples should have been filtered out"
