import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

import chz
from examples.experimental.harbor.harbor_agent_function import run
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

from miles.tinker.client.server import SessionServer
from miles.tinker.client.session import ChatSession
from miles.tinker.client.trajectory import turns_to_trajectory


@dataclass
class HarborEnv(Env):
    task_id: str
    agent_name: str
    verdict: dict | None = None

    async def initial_observation(self):
        raise NotImplementedError("Harbor drives this environment through the OAI adapter")

    async def step(self, action, *, extra=None):
        raise NotImplementedError("Harbor drives this environment through the OAI adapter")


@dataclass
class HarborGroup(EnvGroupBuilder):
    task_id: str
    group_size: int
    agent_name: str

    async def make_envs(self):
        return [HarborEnv(self.task_id, self.agent_name) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        results = []
        for env in env_group:
            assert env.verdict is not None
            reward = float(env.verdict["reward"])
            assert math.isfinite(reward)
            results.append((reward, {f"exit_status/{env.verdict['exit_status']}": 1.0}))
        return results

    def logging_tags(self):
        return ["harbor", self.agent_name, self.task_id]


class HarborDataset(RLDataset):
    def __init__(self, task_ids, groups_per_batch, group_size, agent_name, epochs):
        if not task_ids or min(groups_per_batch, group_size, epochs) < 1:
            raise ValueError("tasks must be nonempty; batch size, group size and epochs must be positive")
        self.task_ids = task_ids
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.agent_name = agent_name
        self.epochs = epochs

    def get_batch(self, index):
        start = index * self.groups_per_batch
        return [HarborGroup(self.task_ids[i % len(self.task_ids)], self.group_size, self.agent_name)
                for i in range(start, min(start + self.groups_per_batch, len(self.task_ids) * self.epochs))]

    def __len__(self):
        return math.ceil(len(self.task_ids) * self.epochs / self.groups_per_batch)


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    tasks_dir: str
    groups_per_batch: int = 4
    group_size: int = 4
    agent_name: str = "terminus-2"
    epochs: int = 1

    async def __call__(self):
        if self.agent_name == "claude-code":
            raise ValueError("this adapter supports OpenAI chat completions, not the Anthropic API")
        task_ids = sorted(path.name for path in Path(self.tasks_dir).iterdir() if (path / "task.toml").is_file())
        return HarborDataset(task_ids, self.groups_per_batch, self.group_size, self.agent_name, self.epochs), None


@dataclass(frozen=True)
class SessionRolloutStrategy(RolloutStrategy):
    renderer_name: str
    advertised_host: str = "127.0.0.1"
    listen_host: str = "127.0.0.1"
    max_parallel_trials_per_group: int = 4
    max_datum_tokens: int = 32768
    max_turns: int = 512

    async def execute(self, env_group_builder, policy):
        assert isinstance(policy, TinkerTokenCompleter)
        if min(self.max_parallel_trials_per_group, self.max_datum_tokens, self.max_turns) < 1:
            raise ValueError("trial concurrency, token budget and turn limit must be positive")
        tokenizer = await asyncio.to_thread(policy.sampling_client.get_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        envs = await env_group_builder.make_envs()
        server = SessionServer()
        semaphore = asyncio.Semaphore(self.max_parallel_trials_per_group)
        async with server.serve(self.listen_host) as port:
            async def one(env):
                async with semaphore:
                    session = ChatSession(policy, renderer, self.max_datum_tokens, self.max_turns)
                    async with server.session(session) as path:
                        env.verdict = await run(
                            base_url=f"http://{self.advertised_host}:{port}{path}",
                            prompt=None,
                            request_kwargs={"max_tokens": policy.max_tokens, "temperature": policy.temperature},
                            metadata={"instance_id": env.task_id, "agent_name": env.agent_name,
                                      "max_seq_len": self.max_datum_tokens + 1},
                        )
                    if env.verdict["exit_status"] == "AgentError":
                        raise RuntimeError(f"Harbor trial {env.task_id} failed: {env.verdict}")
                    return turns_to_trajectory(session.trace.turns)

            # An infrastructure failure invalidates the group, rather than becoming a zero reward.
            async with asyncio.TaskGroup() as tasks:
                pending = [tasks.create_task(one(env)) for env in envs]
        return RolloutResult(trajectories=[task.result() for task in pending], envs=envs, errors=[])
