import os
import pickle

import httpx
import pytest
from examples.multi_lora.harbor_tinker import harbor_env, run_harbor_tinker
from tests.e2e.lora.tinker_client.session_cases import make_session
from tinker_cookbook.rl.data_processing import trajectory_to_data


@pytest.mark.asyncio
async def test_harbor_http_turns_become_unmodified_training_data(monkeypatch):
    policy = make_session().policy
    policy.sampling_client.get_tokenizer = lambda: make_session().renderer.tokenizer
    calls = []
    async def trial(base_url, prompt, request_kwargs, metadata):
        calls.append(base_url)
        async with httpx.AsyncClient() as http:
            for text in ["original", "compacted", "compacted"]:
                response = await http.post(f"{base_url}/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": text}], **request_kwargs,
                })
                response.raise_for_status()
        return {"reward": float(len(calls) % 2), "exit_status": "Submitted"}
    monkeypatch.setattr(harbor_env, "run", trial)
    strategy = harbor_env.SessionRolloutStrategy(renderer_name="role_colon", max_parallel_trials_per_group=1)
    strategy = pickle.loads(pickle.dumps(strategy))
    group = harbor_env.HarborGroup("task", 2, "terminus-2")
    outcome = await strategy.execute(group, policy)
    assert len(set(calls)) == 2
    rewards = await group.compute_group_rewards(outcome.trajectories, outcome.envs)
    assert [reward for reward, _ in rewards] == [1, 0]
    for trajectory in outcome.trajectories:
        datums = trajectory_to_data(trajectory, traj_advantage=1.0)
        assert len(datums) == 3
        assert [transition.ac.tokens for transition in trajectory.transitions] == [[1000]] * 3
        assert [datum.loss_fn_inputs["target_tokens"].data[-1] for datum in datums] == [1000] * 3
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.ConnectError):
            await http.post(f"{calls[0]}/v1/chat/completions", json={})


@pytest.mark.asyncio
async def test_infrastructure_failure_cannot_become_a_zero_reward(monkeypatch):
    policy = make_session().policy
    policy.sampling_client.get_tokenizer = lambda: make_session().renderer.tokenizer
    async def trial(**kwargs):
        return {"reward": 0, "exit_status": "AgentError"}
    monkeypatch.setattr(harbor_env, "run", trial)
    strategy = harbor_env.SessionRolloutStrategy(renderer_name="role_colon")
    with pytest.raises(ExceptionGroup, match="TaskGroup") as caught:
        await strategy.execute(harbor_env.HarborGroup("task", 1, "terminus-2"), policy)
    assert "Harbor trial task failed" in str(caught.value.exceptions[0])


def test_explicit_recipe_config_controls_sdk_and_trial_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "tml-ambient")
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/wrong/tasks")
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    observed = []
    async def train(config):
        observed.append((os.environ["TINKER_API_KEY"], os.environ["HARBOR_TASKS_DIR"], config))
    monkeypatch.setattr(run_harbor_tinker.train, "main", train)
    run_harbor_tinker.main(run_harbor_tinker.HarborTinkerConfig(
        gateway="http://gateway", model_name="model", renderer_name="role_colon",
        tasks_dir=str(tmp_path), api_key="tml-explicit",
    ))
    key, tasks, config = observed[0]
    assert key == "tml-explicit"
    assert tasks == str(tmp_path)
    assert config.base_url == "http://gateway"
    assert config.ttl_seconds is None
