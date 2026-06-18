import pytest

from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(["--group-rm", "--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
            id="group_rm_enabled",
        ),
    ],
    indirect=True,
)
def test_group_rm_rewards_set(rollout_env):
    env = rollout_env
    out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    rewards = [sample.reward for group in out.samples for sample in group]
    assert all(r in (0, 1) for r in rewards)
