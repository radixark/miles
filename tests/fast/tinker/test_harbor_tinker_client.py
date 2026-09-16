"""The cookbook plug-ins in examples/multi_lora/harbor_tinker/harbor_env.py against the real collector app.

Skipped when tinker-cookbook is not installed (it is a client-side, example-scoped dependency). Reused, not reimplemented:
the conftest ``service`` fixture (running TinkerService over ``harness.FakeBackend``), ``oai_fakes`` (character tokenizer,
sampler META.json) and ``build_app_with_collector`` behind an httpx ASGI transport, so the strategy talks to the gateway
exactly as it would over the network.
"""

import asyncio
from dataclasses import dataclass

import httpx
import pytest
from tests.fast.tinker.oai_fakes import SAMPLER, TENANT, FakeTokenizer, write_sampler

from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.oai_routes import build_app_with_collector

pytest.importorskip("tinker_cookbook")
from examples.multi_lora.harbor_tinker.harbor_env import (  # noqa: E402
    HarborDatasetBuilder,
    HarborEnv,
    HarborGroup,
    SessionRolloutStrategy,
    turns_to_trajectory,
    verdict_reward,
)
from tinker_cookbook.exceptions import AllTrajectoriesFailedError  # noqa: E402
from tinker_cookbook.rl.data_processing import trajectory_to_data  # noqa: E402

GATEWAY = "http://gateway"
ENGINE_OUTPUT = [1, 2]  # what harness.FakeBackend.sample returns for every request


def _turn(input_ids, output_ids, finish_reason="stop"):
    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "logprobs": [-0.5] * len(output_ids),
        "finish_reason": finish_reason,
        "created_at": 0.0,
    }


def test_turns_to_trajectory_is_a_cookbook_transition_per_turn():
    """Each turn is Transition(ob=input_ids, ac=output_ids+logprobs); the last one carries episode_done and the stop metric."""
    trajectory = turns_to_trajectory([_turn([1, 2, 3], [4, 5]), _turn([1, 2, 3, 4, 5, 6], [7], "length")])
    first, last = trajectory.transitions
    assert first.ob.to_ints() == [1, 2, 3] and first.ac.tokens == [4, 5] and first.ac.logprobs == [-0.5, -0.5]
    assert (first.episode_done, last.episode_done) == (False, True)
    assert last.ac.stop_reason == "length" and last.metrics == {"stop/max_tokens": 1.0}
    assert trajectory.final_ob.to_ints() == [1, 2, 3, 4, 5, 6, 7] and trajectory.stop_reason == "max_tokens"
    with pytest.raises(ValueError):
        turns_to_trajectory([])


def test_chained_turns_merge_and_a_broken_prefix_splits():
    """trajectory_to_data (cookbook) yields one Datum when every turn extends the previous prompt+output, else one per turn."""
    chained = turns_to_trajectory([_turn([1, 2, 3], [4, 5]), _turn([1, 2, 3, 4, 5, 6], [7])])
    (datum,) = trajectory_to_data(chained, traj_advantage=1.0)
    assert datum.model_input.to_ints() == [1, 2, 3, 4, 5, 6]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4, 5, 6, 7]
    assert datum.loss_fn_inputs["mask"].tolist() == [0, 0, 1, 1, 0, 1]

    broken = turns_to_trajectory([_turn([1, 2, 3], [4, 5]), _turn([1, 2, 9, 6], [7])])
    assert len(trajectory_to_data(broken, traj_advantage=1.0)) == 2


def test_verdict_reward_and_group_rewards():
    """compute_group_rewards reads each env's verdict: reward, one-hot exit_status, numeric agent metrics; no verdict scores 0."""
    verdict = {
        "reward": 1.0,
        "exit_status": "Submitted",
        "agent_metrics": {"turns": 3, "agent_run_time": 12.5, "note": "x"},
    }
    assert verdict_reward(verdict) == (
        1.0,
        {"exit_status/Submitted": 1, "agent/turns": 3, "agent/agent_run_time": 12.5},
    )
    assert verdict_reward(None) == (0.0, {"exit_status/NoVerdict": 1})

    group = HarborGroup("fix-git", group_size=2)
    envs = asyncio.run(group.make_envs())
    envs[0].verdict = verdict
    rewards = asyncio.run(group.compute_group_rewards([], envs))
    assert [reward for reward, _ in rewards] == [1.0, 0.0]
    assert group.logging_tags() == ["harbor", "terminus-2", "fix-git"]


def test_dataset_builder_lists_task_dirs_and_wraps_batches(tmp_path):
    """Task ids are the directories with a task.toml; batches slice the sorted list and wrap around."""
    for name in ("fix-git", "bn-fit-modify", "not-a-task"):
        (tmp_path / name).mkdir()
    (tmp_path / "fix-git" / "task.toml").write_text("")
    (tmp_path / "bn-fit-modify" / "task.toml").write_text("")
    (tmp_path / "loose-file").write_text("")

    dataset, test_split = asyncio.run(
        HarborDatasetBuilder(tasks_dir=str(tmp_path), groups_per_batch=3, group_size=2)()
    )
    assert test_split is None and len(dataset) == 1
    batch = dataset.get_batch(0)
    assert [group.task_id for group in batch] == ["bn-fit-modify", "fix-git", "bn-fit-modify"]
    assert all(group.group_size == 2 for group in batch)
    with pytest.raises(ValueError):
        asyncio.run(HarborDatasetBuilder(tasks_dir=str(tmp_path / "not-a-task"))())


@dataclass
class FakeSamplingClient:
    _sampling_session_id: str


@dataclass
class FakePolicy:
    sampling_client: FakeSamplingClient


@pytest.fixture
async def gateway(service, tmp_path):
    """The real collector app over the running FakeBackend service, reachable through an ASGI transport."""
    write_sampler(tmp_path, service.config.base_model)
    service.config.vocab_size = 1000
    collector = TrajectoryCollector(service, FakeTokenizer(), session_ttl_s=600.0, chat_template_kwargs=None)
    tinker_session = service.create_session(TENANT)
    sampling_session_id = service.create_sampling_session(
        TENANT, {"session_id": tinker_session, "sampling_session_seq_id": 0, "model_path": SAMPLER}
    )
    transport = httpx.ASGITransport(app=build_app_with_collector(service, collector))
    return transport, collector, FakePolicy(FakeSamplingClient(sampling_session_id))


def _fake_trial(transport: httpx.ASGITransport, turns: int, reward: float = 1.0):
    """Stand-in for harbor_agent_function.run: the agent talks to base_url + /v1 with the dummy key, `turns` times."""

    async def run(base_url, prompt, request_kwargs, metadata):
        async with httpx.AsyncClient(transport=transport, base_url=f"{base_url}/v1") as agent:
            messages = [{"role": "user", "content": metadata["instance_id"]}]
            for _ in range(turns):
                response = await agent.post(
                    "/chat/completions",
                    json={"model": "openai/model", "messages": messages, **request_kwargs},
                    headers={"Authorization": "Bearer dummy"},
                )
                response.raise_for_status()
                messages.append(response.json()["choices"][0]["message"])
                messages.append({"role": "user", "content": "ok"})
        return {
            "reward": reward,
            "exit_status": "Submitted",
            "eval_report": {"reward": reward},
            "agent_metrics": {"turns": turns},
        }

    return run


async def test_strategy_runs_trials_through_recorded_sessions(gateway):
    """bind → trial (two recorded chats) → GET turns → DELETE; the Trajectory holds exactly the engine's ids and the env holds the verdict."""
    transport, collector, policy = gateway
    strategy = SessionRolloutStrategy(
        GATEWAY, TENANT, max_tokens=8, run_trial=_fake_trial(transport, turns=2), transport=transport
    )
    group = HarborGroup("fix-git", group_size=2)
    result = await strategy.execute(group, policy)

    assert result.errors == [] and len(result.trajectories) == len(result.envs) == 2
    trajectory = result.trajectories[0]
    assert len(trajectory.transitions) == 2
    first = trajectory.transitions[0]
    assert first.ob.to_ints() == [2, *(ord(c) for c in "fix-git"), 9]  # user marker, the task id, generation prompt
    assert first.ac.tokens == ENGINE_OUTPUT and first.ac.logprobs == [0.0, 0.0]
    second_ob = trajectory.transitions[1].ob.to_ints()
    assert second_ob[: len(first.ob.to_ints()) + 2] != first.ob.to_ints() + ENGINE_OUTPUT
    assert (
        len(trajectory_to_data(trajectory, traj_advantage=1.0)) == 2
    )  # re-rendered history, no prefix: one Datum per turn

    env = result.envs[0]
    assert isinstance(env, HarborEnv) and env.verdict["reward"] == 1.0 and env.verdict["model_path"] == SAMPLER
    assert collector.sessions == {}  # every session was deleted after export
    assert (await group.compute_group_rewards(result.trajectories, result.envs))[0][0] == 1.0


async def test_failed_trials_become_rollout_errors(gateway):
    """A trial that raises or records no turn is a RolloutError; the survivors keep their order; all failing raises."""
    transport, collector, policy = gateway
    calls = {"n": 0}

    async def flaky(base_url, prompt, request_kwargs, metadata):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("sandbox exploded")
        return await _fake_trial(transport, turns=1)(base_url, prompt, request_kwargs, metadata)

    strategy = SessionRolloutStrategy(
        GATEWAY, TENANT, max_tokens=8, concurrency=1, run_trial=flaky, transport=transport
    )
    result = await strategy.execute(HarborGroup("fix-git", group_size=2), policy)
    assert [error.error_type for error in result.errors] == ["RuntimeError"]
    assert len(result.trajectories) == len(result.envs) == 1
    assert collector.sessions == {}

    async def silent(base_url, prompt, request_kwargs, metadata):
        return {"reward": 0.0, "exit_status": "AgentError", "eval_report": {}, "agent_metrics": {}}

    strategy = SessionRolloutStrategy(GATEWAY, TENANT, max_tokens=8, run_trial=silent, transport=transport)
    with pytest.raises(AllTrajectoriesFailedError):
        await strategy.execute(HarborGroup("fix-git", group_size=1), policy)
