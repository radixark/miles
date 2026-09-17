"""tinker-cookbook plug-ins for Harbor tasks sampled through recorded gateway sessions: dataset / group / env types, turns → Trajectory, and SessionRolloutStrategy (bind by the policy's Tinker sampler → harbor_agent_function.run → export → delete); needs pip install tinker-cookbook."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz
import httpx
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import (
    STOP_METRIC_PREFIX,
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    InitialObservationOverflow,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    RolloutError,
    StepResult,
    StopCondition,
    StopReason,
    Trajectory,
    Transition,
)

import tinker

logger = logging.getLogger(__name__)

# what a Turn's finish_reason ("stop" | "length", Tinker's StopReason) means in cookbook StopReason terms
_STOP_REASONS = {"stop": StopReason.COMPLETED, "length": StopReason.MAX_TOKENS}

RunTrial = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class HarborEnv(Env):
    """One Harbor task instance for one trajectory: task_id, harness name, and the Harbor verdict once the trial ran."""

    task_id: str
    agent_name: str = "terminus-2"
    verdict: dict[str, Any] | None = None  # harbor_agent_function.run's return value

    async def initial_observation(self) -> tuple[Observation, StopCondition] | InitialObservationOverflow:
        """Never driven by the runner: the Harbor agent samples through the gateway session, so this raises if called."""
        raise NotImplementedError("HarborEnv is driven by SessionRolloutStrategy, not by the cookbook rollout loop")

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Never driven by the runner (see initial_observation)."""
        raise NotImplementedError("HarborEnv is driven by SessionRolloutStrategy, not by the cookbook rollout loop")


def verdict_reward(verdict: dict[str, Any] | None) -> tuple[float, Metrics]:
    """(reward, metrics) from a Harbor verdict: the verifier's reward, a one-hot exit_status, and the numeric agent_metrics."""
    if not verdict:
        return 0.0, {"exit_status/NoVerdict": 1}
    metrics: Metrics = {f"exit_status/{verdict.get('exit_status', 'AgentError')}": 1}
    for key, value in (verdict.get("agent_metrics") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[f"agent/{key}"] = value
    return float(verdict.get("reward", 0.0)), metrics


class HarborGroup(EnvGroupBuilder):
    """group_size HarborEnvs for one task; compute_group_rewards reads each env's Harbor verdict (cookbook's do_group_rollout calls it)."""

    def __init__(self, task_id: str, group_size: int, agent_name: str = "terminus-2") -> None:
        """Remember the task id, how many trajectories to sample for it, and which harness runs it."""
        self.task_id = task_id
        self.group_size = group_size
        self.agent_name = agent_name

    async def make_envs(self) -> Sequence[Env]:
        """group_size fresh HarborEnv instances for this task."""
        return [HarborEnv(task_id=self.task_id, agent_name=self.agent_name) for _ in range(self.group_size)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Per trajectory: (verdict["reward"], {exit_status flags, agent_metrics}); a missing verdict scores 0."""
        return [verdict_reward(getattr(env, "verdict", None)) for env in env_group]

    def logging_tags(self) -> list[str]:
        """Aggregate metrics under the harness name and the task."""
        return ["harbor", self.agent_name, self.task_id]


class HarborDataset(RLDataset):
    """Batches of HarborGroups over the task directories (a Terminal-Bench-2 checkout works as-is); no cookbook equivalent."""

    def __init__(
        self, task_ids: list[str], groups_per_batch: int, group_size: int, agent_name: str, epochs: int = 1
    ) -> None:
        """Keep the task list and batch shape; batches are consecutive slices of the task list, wrapping around for `epochs` passes."""
        if not task_ids:
            raise ValueError("HarborDataset needs at least one task")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.task_ids = list(task_ids)
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.agent_name = agent_name
        self.epochs = epochs

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """groups_per_batch builders for batch `index`."""
        start = index * self.groups_per_batch
        return [
            HarborGroup(self.task_ids[(start + offset) % len(self.task_ids)], self.group_size, self.agent_name)
            for offset in range(self.groups_per_batch)
        ]

    def __len__(self) -> int:
        """Number of batches in `epochs` passes over the task list (the cookbook loop runs exactly len(dataset) steps unless max_steps is lower)."""
        return math.ceil(len(self.task_ids) * self.epochs / self.groups_per_batch)


def list_task_ids(tasks_dir: str) -> list[str]:
    """Task ids are the directories under tasks_dir holding a task.toml (Harbor's task layout), sorted for a stable order."""
    root = Path(tasks_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"tasks_dir {tasks_dir!r} is not a directory")
    return sorted(path.name for path in root.iterdir() if (path / "task.toml").is_file())


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    """Config-side builder: tasks_dir → (HarborDataset, None); task ids are the directories holding a task.toml. Set HARBOR_TASKS_DIR to the same directory for harbor_agent_function."""

    tasks_dir: str
    groups_per_batch: int = 4
    group_size: int = 4
    agent_name: str = "terminus-2"
    epochs: int = 1  # passes over the task list; the cookbook runs len(dataset) steps

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """List the task directories under tasks_dir (real directories: harbor_agent_function rejects symlinks that resolve outside HARBOR_TASKS_DIR) and wrap them in a HarborDataset; no test split."""
        task_ids = list_task_ids(self.tasks_dir)
        if not task_ids:
            raise ValueError(f"no task directory with a task.toml under {self.tasks_dir!r}")
        return HarborDataset(task_ids, self.groups_per_batch, self.group_size, self.agent_name, self.epochs), None


def truncate_turns(turns: list[dict[str, Any]], max_tokens: int | None) -> list[dict[str, Any]]:
    """Keep the leading turns whose prompt + output fit the gateway's per-datum cap (serve_tinker.py: min(model max_position_embeddings, --max-tokens-per-gpu rounded to the pad size); 8192 with the example launcher's defaults, so raise --max-tokens-per-gpu for agent contexts); prompts grow every turn, so the first over-long turn ends the trainable part of the trajectory (the Harbor agent's own max_seq_len check counts tokens approximately)."""
    if max_tokens is None:
        return turns
    kept = []
    for turn in turns:
        if len(turn["input_ids"]) + len(turn["output_ids"]) > max_tokens:
            break
        kept.append(turn)
    return kept


def turns_to_trajectory(turns: list[dict[str, Any]]) -> Trajectory:
    """GET /oai/sessions/{sid} turns → Trajectory: each turn is Transition(ob=ModelInput.from_ints(input_ids), ac=TokensWithLogprobs(output_ids, logprobs, finish_reason), reward=0.0, episode_done on the last); final_ob = last input_ids + output_ids; stop_reason from the last finish_reason. The only data shaping of ours; the cookbook's only Trajectory producer is its own run_rollout."""
    if not turns:
        raise ValueError("the session recorded no turns; nothing to train on")
    last_index = len(turns) - 1
    transitions = []
    for index, turn in enumerate(turns):
        stop_reason = _STOP_REASONS.get(turn["finish_reason"], StopReason.COMPLETED)
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(list(turn["input_ids"])),
                ac=TokensWithLogprobs(
                    tokens=list(turn["output_ids"]),
                    maybe_logprobs=list(turn["logprobs"]),
                    stop_reason=turn["finish_reason"],
                ),
                reward=0.0,
                episode_done=index == last_index,
                metrics={f"{STOP_METRIC_PREFIX}{stop_reason}": 1.0} if index == last_index else {},
            )
        )
    last = turns[-1]
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints(list(last["input_ids"]) + list(last["output_ids"])),
        stop_reason=str(_STOP_REASONS.get(last["finish_reason"], StopReason.COMPLETED)),
    )


def sampling_session_id_of(policy: TokenCompleter) -> str:
    """The Tinker sampling session behind the cookbook's TinkerTokenCompleter (the SDK keeps it as SamplingClient._sampling_session_id); the gateway resolves it to the sampler path with get_sampler."""
    sampling_client = getattr(policy, "sampling_client", None)
    sampling_session_id = getattr(sampling_client, "_sampling_session_id", None) or getattr(
        sampling_client, "sampling_session_id", None
    )
    if not sampling_session_id:
        raise TypeError("SessionRolloutStrategy needs a TinkerTokenCompleter over a tinker.SamplingClient")
    return str(sampling_session_id)


def _default_run_trial() -> RunTrial:
    """harbor_agent_function.run, imported when a trial actually runs: it pulls in Harbor and the sandbox SDK, which the data path does not need."""
    from examples.experimental.harbor.harbor_agent_function import run

    return run


@dataclass(frozen=True)
class SessionRolloutStrategy(RolloutStrategy):
    """Drop-in for cookbook's FailFast via Config.rollout_error_tolerance: per HarborEnv, bind a recorded gateway session to the policy's sampler version, run the Harbor agent against it, rebuild the Trajectory from the recorded turns."""

    gateway_url: str
    api_key: str
    concurrency: int = 16
    max_seq_len: int = 65536
    max_tokens: int = 8192
    temperature: float = 1.0
    http_timeout_s: float = 60.0
    max_datum_tokens: int | None = (
        32768  # must not exceed the gateway's per-datum cap (see truncate_turns); longer turns are cut off the trajectory
    )
    record_path: str | None = None  # append one JSON line per trajectory (task, turns, token counts, reward) when set
    # test seams: the trial runner (default harbor_agent_function.run) and an httpx transport (default: the network)
    run_trial: RunTrial | None = field(default=None, compare=False, repr=False)
    transport: httpx.AsyncBaseTransport | None = field(default=None, compare=False, repr=False)

    async def execute(self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter) -> RolloutResult:
        """make_envs, bind every env's session to sampling_session_id_of(policy), run the trials under a semaphore, return RolloutResult(trajectories, envs, errors); a failed trial becomes a cookbook RolloutError, not an exception."""
        envs = await env_group_builder.make_envs()
        sampling_session_id = sampling_session_id_of(policy)
        semaphore = asyncio.Semaphore(self.concurrency)
        async with self._client() as http:

            async def one(env: HarborEnv) -> Trajectory:
                async with semaphore:
                    return await self.run_one(env, sampling_session_id, http)

            outcomes = await asyncio.gather(*(one(env) for env in envs), return_exceptions=True)
        trajectories: list[Trajectory] = []
        survivors: list[Env] = []
        errors: list[RolloutError] = []
        for env, outcome in zip(envs, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("Harbor trial for %s failed: %s: %s", env.task_id, type(outcome).__name__, outcome)
                errors.append(RolloutError(error_type=type(outcome).__name__, error_message=str(outcome)))
            else:
                trajectories.append(outcome)
                survivors.append(env)
        if not trajectories:
            raise AllTrajectoriesFailedError(
                f"every trial of {getattr(env_group_builder, 'task_id', '?')} failed: {errors}"
            )
        return RolloutResult(trajectories=trajectories, envs=survivors, errors=errors)

    async def run_one(self, env: HarborEnv, sampling_session_id: str, http: httpx.AsyncClient) -> Trajectory:
        """POST /oai/sessions/{sid} {sampling_session_id, max_datum_tokens} → harbor_agent_function.run(base_url=…/oai/sessions/{sid}, metadata={instance_id, agent_name, max_seq_len}, request_kwargs={max_tokens, temperature}) → env.verdict → GET turns → DELETE → turns_to_trajectory."""
        session_id = f"harbor-{uuid.uuid4().hex}"
        bind_body: dict[str, Any] = {"sampling_session_id": sampling_session_id}
        if self.max_datum_tokens is not None:
            bind_body["max_datum_tokens"] = (
                self.max_datum_tokens
            )  # the TITO chain budget, same cap truncate_turns enforces
        bound = await http.post(f"/oai/sessions/{session_id}", json=bind_body)
        bound.raise_for_status()
        try:
            run_trial = self.run_trial or _default_run_trial()
            verdict = await run_trial(
                base_url=f"{self.gateway_url}/oai/sessions/{session_id}",
                prompt=None,
                request_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
                metadata={"instance_id": env.task_id, "agent_name": env.agent_name, "max_seq_len": self.max_seq_len},
            )
            env.verdict = {**verdict, "model_path": bound.json().get("model_path")}
            exported = await http.get(f"/oai/sessions/{session_id}")
            exported.raise_for_status()
        finally:
            try:
                await http.delete(f"/oai/sessions/{session_id}")
            except httpx.HTTPError as error:  # the gateway's TTL sweep is the fallback
                logger.warning("could not delete session %s: %s", session_id, error)
        turns = exported.json()["turns"]
        kept = truncate_turns(turns, self.max_datum_tokens)
        if len(kept) < len(turns):
            logger.warning(
                "%s: %d of %d turns exceed %d tokens and are left out of the trajectory",
                env.task_id,
                len(turns) - len(kept),
                len(turns),
                self.max_datum_tokens,
            )
        self._record(env, session_id, turns, len(turns) - len(kept))
        return turns_to_trajectory(kept)

    def _record(self, env: HarborEnv, session_id: str, turns: list[dict[str, Any]], dropped: int = 0) -> None:
        """Experiment log: one JSON line per trajectory with the task, turn count, per-turn prompt/output token counts, final sequence length, turns dropped by the datum cap, reward and exit_status."""
        if not self.record_path:
            return
        verdict = env.verdict or {}
        line = {
            "time": time.time(),
            "task": env.task_id,
            "session": session_id,
            "turns": len(turns),
            "prompt_tokens": [len(turn["input_ids"]) for turn in turns],
            "output_tokens": [len(turn["output_ids"]) for turn in turns],
            "final_len": (len(turns[-1]["input_ids"]) + len(turns[-1]["output_ids"])) if turns else 0,
            "dropped_turns": dropped,
            "reward": verdict.get("reward"),
            "exit_status": verdict.get("exit_status"),
            "model_path": verdict.get("model_path"),
        }
        with open(self.record_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def _client(self) -> httpx.AsyncClient:
        """One HTTP client per group: the gateway URL, the tenant's bearer, and the optional test transport."""
        return httpx.AsyncClient(
            base_url=self.gateway_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.http_timeout_s,
            transport=self.transport,
        )
