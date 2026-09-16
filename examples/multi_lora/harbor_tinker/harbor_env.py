"""tinker-cookbook RL plug-ins for Harbor tasks sampled through the gateway's recorded sessions.

Skeleton: classes document what they will do; bodies land in follow-up commits.

Reused, not reimplemented:
- cookbook ``rl/``: ``Env`` / ``EnvGroupBuilder`` / ``RLDataset`` / ``RLDatasetBuilder`` (types.py), ``RolloutStrategy`` /
  ``RolloutResult`` / ``RolloutError`` (rollout_strategy.py, types.py), ``do_group_rollout`` which calls the strategy and
  ``compute_group_rewards``, and ``train.py`` for the whole loop. There is no generic list-of-tasks RLDataset in the cookbook
  (problem_env.py only has ProblemEnv/ProblemGroupBuilder for single-turn Q&A), so HarborRLDataset is ours.
- miles: ``examples/experimental/harbor/harbor_agent_function.run`` (Harbor TrialConfig, sandbox provider from
  HARBOR_ENV_TYPE / E2B_* env vars, verdict → {reward, exit_status, eval_report, agent_metrics}),
  ``miles.rollout.agentic.session.resolve_session_url``, ``miles.rollout.agentic.credentials`` (preflight_sdk, PROVIDER_CREDENTIALS).
- gateway: ``GET /api/v1/samplers/{sampling_session_id}`` returns the sampler ``model_path`` behind a Tinker SamplingClient.

The one twist versus a cookbook env: the Harbor agent samples through the gateway's OpenAI-compatible session, not
through the runner's ``policy(ob)`` call, so ``SessionRolloutStrategy`` runs the trial and rebuilds the ``Trajectory``
from the recorded turns instead of stepping the env.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import chz
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    InitialObservationOverflow,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    StopCondition,
    Trajectory,
)


@dataclass
class HarborEnv(Env):
    """One Harbor task instance for one trajectory: task_id, harness name, and the Harbor verdict once the trial ran."""

    task_id: str
    agent_name: str = "terminus-2"
    verdict: dict[str, Any] | None = None  # harbor_agent_function.run's return value

    async def initial_observation(self) -> tuple[Observation, StopCondition] | InitialObservationOverflow:
        """Never driven by the runner: the Harbor agent samples through the gateway session, so this raises if called."""
        raise NotImplementedError

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Never driven by the runner (see initial_observation)."""
        raise NotImplementedError


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """group_size HarborEnvs for one task; compute_group_rewards reads each env's Harbor verdict (cookbook's do_group_rollout calls it)."""

    def __init__(self, task_id: str, group_size: int, agent_name: str = "terminus-2") -> None:
        """Remember the task id, how many trajectories to sample for it, and which harness runs it."""
        self.task_id = task_id
        self.group_size = group_size
        self.agent_name = agent_name

    async def make_envs(self) -> Sequence[Env]:
        """group_size fresh HarborEnv instances for this task."""
        raise NotImplementedError

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Per trajectory: (verdict["reward"], {exit_status flags, agent_metrics}); a missing verdict scores 0."""
        raise NotImplementedError

    def logging_tags(self) -> list[str]:
        """['harbor', self.agent_name] so cookbook metrics aggregate per harness."""
        raise NotImplementedError


class HarborRLDataset(RLDataset):
    """Batches of HarborEnvGroupBuilders over the task directories (a Terminal-Bench-2 checkout works as-is); no cookbook equivalent."""

    def __init__(self, task_ids: list[str], groups_per_batch: int, group_size: int, agent_name: str) -> None:
        """Keep the task list and batch shape; batches are consecutive slices of the (shuffled) task list."""
        self.task_ids = task_ids
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.agent_name = agent_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """groups_per_batch builders for batch `index`, wrapping around the task list."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of batches before the task list wraps."""
        raise NotImplementedError


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    """Config-side builder: tasks_dir → (HarborRLDataset, None); task ids are the Harbor task directories."""

    tasks_dir: str
    groups_per_batch: int = 4
    group_size: int = 4
    agent_name: str = "terminus-2"

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """List the task directories under tasks_dir and wrap them in a HarborRLDataset; no test split."""
        raise NotImplementedError


@dataclass(frozen=True)
class SessionRolloutStrategy(RolloutStrategy):
    """Drop-in for cookbook's FailFast via Config.rollout_error_tolerance: per HarborEnv, bind a recorded gateway session to the policy's sampler version, run the Harbor agent against it, rebuild the Trajectory from the recorded turns."""

    gateway_url: str
    tinker_api_key: str
    concurrency: int = 16
    max_seq_len: int = 65536
    request_kwargs: dict[str, Any] = field(
        default_factory=dict
    )  # sampling params handed to the harness (max_tokens, temperature)

    async def execute(self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter) -> RolloutResult:
        """make_envs, run every env under a semaphore, return RolloutResult(trajectories, envs, errors); a failed trial becomes a cookbook RolloutError, not an exception."""
        raise NotImplementedError

    async def _run_one(self, env: HarborEnv, sampler_model_path: str) -> Trajectory:
        """SessionClient.bind → harbor_agent_function.run(base_url=agent_base_url, metadata={instance_id, agent_name, max_seq_len}, request_kwargs) → env.verdict → SessionClient.trajectory → SessionClient.delete."""
        raise NotImplementedError

    def _sampler_model_path(self, policy: TokenCompleter) -> str:
        """The tinker://…/sampler_weights/… path behind the policy: policy.sampling_client._sampling_session_id → the gateway's existing GET /api/v1/samplers/{id}['model_path']."""
        raise NotImplementedError
