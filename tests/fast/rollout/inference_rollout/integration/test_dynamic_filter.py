from contextlib import nullcontext

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.misc import function_registry


@pytest.mark.parametrize(
    "rollout_env,use_filter,expect_all_correct",
    [
        pytest.param(
            integration_env_config(["--rollout-batch-size", "4"], data_rows=MIXED_DATA_ROWS),
            False,
            False,
            id="no_filter",
        ),
        pytest.param(
            integration_env_config(
                ["--rollout-batch-size", "3", "--dynamic-sampling-filter-path", "test:filter_by_reward"],
                data_rows=MIXED_DATA_ROWS,
            ),
            True,
            True,
            id="with_filter",
        ),
    ],
    indirect=["rollout_env"],
)
def test_filter_effect(rollout_env, use_filter, expect_all_correct):
    env = rollout_env
    ctx = function_registry.temporary("test:filter_by_reward", filter_by_reward) if use_filter else nullcontext()

    with ctx:
        out = load_and_call_train(env.args, env.data_source)

    rewards = {group[0].reward for group in out.samples}
    if expect_all_correct:
        assert rewards == {1}, "Filter should keep only correct samples"
    else:
        assert 0 in rewards, "Without filter, incorrect samples should be present"
