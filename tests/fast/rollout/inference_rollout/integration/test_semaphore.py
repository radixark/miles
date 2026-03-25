import pytest

from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train

_DATA_ROWS = [{"input": f"What is 1+{i}?", "label": str(1 + i)} for i in range(10)]
_BASE_ARGV = ["--rollout-batch-size", "4", "--n-samples-per-prompt", "2"]


@pytest.mark.parametrize(
    "rollout_env,expected_range",
    [
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "1"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (1, 1),
            id="limit_1",
        ),
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "999"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (2, 999),
            id="no_limit",
        ),
    ],
    indirect=["rollout_env"],
)
def test_max_concurrent(rollout_env, expected_range):
    env = rollout_env
    load_and_call_train(env.args, env.data_source)
    min_expected, max_expected = expected_range
    assert min_expected <= env.mock_server.max_concurrent <= max_expected
