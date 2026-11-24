import asyncio
import json
import math
import re
import secrets
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import chz
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StopReason


@dataclass
class HarborEnv(Env):
    task_path: Path
    reward: float | None = None

    async def initial_observation(self):
        raise NotImplementedError("Harbor drives the agent through the official proxy")

    async def step(self, action, *, extra=None):
        raise NotImplementedError("Harbor drives the agent through the official proxy")


@dataclass
class HarborGroup(EnvGroupBuilder):
    task_path: Path
    group_size: int

    async def make_envs(self):
        return [HarborEnv(self.task_path) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        assert all(env.reward is not None for env in env_group)
        return [
            (env.reward, {f"stop/{trajectory.stop_reason}": 1.0})
            for trajectory, env in zip(trajectory_group, env_group, strict=True)
        ]

    def logging_tags(self):
        return ["harbor", self.task_path.name]


class HarborDataset(RLDataset):
    def __init__(self, task_paths, groups_per_batch, group_size, epochs):
        if not task_paths or min(groups_per_batch, group_size, epochs) < 1:
            raise ValueError("tasks must be nonempty; batch size, group size and epochs must be positive")
        self.task_paths = task_paths
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.epochs = epochs

    def get_batch(self, index):
        start = index * self.groups_per_batch
        return [
            HarborGroup(self.task_paths[i % len(self.task_paths)], self.group_size)
            for i in range(start, min(start + self.groups_per_batch, len(self.task_paths) * self.epochs))
        ]

    def __len__(self):
        return math.ceil(len(self.task_paths) * self.epochs / self.groups_per_batch)


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    tasks_dir: str
    groups_per_batch: int = 4
    group_size: int = 4
    epochs: int = 1

    async def __call__(self):
        paths = sorted(path.resolve() for path in Path(self.tasks_dir).iterdir() if (path / "task.toml").is_file())
        return HarborDataset(paths, self.groups_per_batch, self.group_size, self.epochs), None


def trial_config(template, task_path, proxy_url, proxy_api_key):
    if "{proxy_url}" not in template:
        raise ValueError("Harbor config must route its agent to {proxy_url}")
    # Escape injected strings for JSON; all other fields retain Harbor's own schema and defaults.
    for name, value in (("proxy_url", proxy_url), ("proxy_api_key", proxy_api_key)):
        template = template.replace("{" + name + "}", json.dumps(value)[1:-1])
    return TrialConfig(task=TaskConfig(path=task_path), **json.loads(template))


def _class_names(cls):
    return {cls.__name__}.union(*map(_class_names, cls.__subclasses__()))


# Harbor picks a subclass by scanning the agent's whole output, e.g. NetworkConnectionError for the
# agent's own failed downloads, so the subclass does not say why the agent exited.
AGENT_EXIT_ERRORS = _class_names(NonZeroAgentExitCodeError)


def trial_outcome(result, reward_key):
    failure = result.exception_info
    if failure is None:
        stop_reason = StopReason.COMPLETED
    elif failure.exception_type == "AgentTimeoutError":
        stop_reason = StopReason.ROLLOUT_TIMEOUT
    elif failure.exception_type == "ContextWindowExceededError" or (
        failure.exception_type in AGENT_EXIT_ERRORS
        # Harbor misses the official proxy's wording; match only the terminal CLI error.
        and re.search(
            r"\nBadRequestError:[^\n]*prompt is too long:\s*\d+ prompt tokens plus \d+ max_tokens "
            r"exceed the model's context window\s*\nstderr: None\Z",
            failure.exception_message,
        )
    ):
        stop_reason = StopReason.CONTEXT_OVERFLOW
    elif failure.exception_type == "OutputTokenExceededError":
        stop_reason = StopReason.MAX_TOKENS
    else:
        raise RuntimeError(f"Harbor trial failed: {failure.exception_type}: {failure.exception_message}")
    verifier = result.verifier_result
    if verifier is None or verifier.rewards is None:
        raise RuntimeError("Harbor trial has no verifier reward")
    reward = float(verifier.rewards[reward_key])
    if not math.isfinite(reward):
        raise ValueError(f"Harbor returned a non-finite reward: {reward}")
    return reward, str(stop_reason)


@chz.chz
class HarborConfig:
    trial_config: str
    reward_key: str = "reward"
    advertised_host: str = "127.0.0.1"
    listen_host: str = "127.0.0.1"
    max_parallel_trials_per_group: int = 4


@dataclass(frozen=True)
class HarborRolloutStrategy(RolloutStrategy):
    renderer_name: str
    trial_template: str
    store_url: str
    run_id: str
    harbor: HarborConfig
    _groups: count = field(default_factory=count, init=False, repr=False)

    async def execute(self, env_group_builder, policy):
        assert isinstance(policy, TinkerTokenCompleter)
        if self.harbor.max_parallel_trials_per_group < 1:
            raise ValueError("trial concurrency must be positive")
        tokenizer = await asyncio.to_thread(policy.sampling_client.get_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        envs = await env_group_builder.make_envs()
        group_idx = next(self._groups)
        semaphore = asyncio.Semaphore(self.harbor.max_parallel_trials_per_group)
        auth_token = secrets.token_urlsafe(32)
        async with serve_proxy(
            renderer,
            policy.sampling_client,
            model_label="policy",
            default_max_tokens=policy.max_tokens,
            host=self.harbor.listen_host,
            auth_token=auth_token,
        ) as port:

            async def one(index, env):
                async with semaphore:
                    config = trial_config(
                        self.trial_template,
                        env.task_path,
                        f"http://{self.harbor.advertised_host}:{port}/r/run/{self.run_id}/group/{group_idx}/traj/{index}",
                        auth_token,
                    )
                    trial = await Trial.create(config)
                    result = await trial.run()
                    env.reward, stop_reason = trial_outcome(result, self.harbor.reward_key)
                    return stop_reason

            async with asyncio.TaskGroup() as tasks:
                trials = [tasks.create_task(one(index, env)) for index, env in enumerate(envs)]
        await flush_capture()
        trajectories = [
            rows_to_trajectory(
                await read_samples(self.store_url, self.run_id, group_idx=group_idx, traj_idx=index),
                trial.result(),
            )
            for index, trial in enumerate(trials)
        ]
        return RolloutResult(trajectories=trajectories, envs=envs, errors=[])
