import pytest

from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train


@pytest.mark.parametrize(
    "rollout_env,expected_seeds",
    [
        pytest.param(
            integration_env_config(
                [
                    "--sglang-enable-deterministic-inference",
                    "--rollout-seed",
                    "42",
                    "--n-samples-per-prompt",
                    "3",
                    "--rollout-batch-size",
                    "1",
                ]
            ),
            {42, 43, 44},
            id="enabled",
        ),
        pytest.param(
            integration_env_config(["--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
            {None},
            id="disabled",
        ),
    ],
    indirect=["rollout_env"],
)
def test_sampling_seeds(rollout_env, expected_seeds):
    env = rollout_env
    load_and_call_train(env.args, env.data_source)

    seeds = {req.get("sampling_params", {}).get("sampling_seed") for req in env.mock_server.request_log}
    assert seeds == expected_seeds
