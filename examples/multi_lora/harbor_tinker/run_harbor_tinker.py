"""One tenant's Harbor × Tinker RL loop: pre-bind sessions, run trials, collect turns, train through the Tinker API.

Skeleton: functions document what they will do; bodies land in follow-up commits.

Usage (stage 1, internal AgentENV sandbox):
    HARBOR_ENV_TYPE=e2b HARBOR_TASKS_DIR=<terminal-bench-2 checkout> \\
    python run_harbor_tinker.py --gateway http://<gateway>:10613 --api-key tml-... --base-model Qwen3-30B-A3B

Stage 0 swaps ``harbor_agent_function.run`` for ``toy_tool_agent.run`` with ``--agent toy``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class LoopArgs:
    """CLI knobs: gateway URL, Tinker key, base model, LoRA rank, tasks dir, tasks per step, samples per task, steps, concurrency, max_seq_len, loss_fn, learning rate, agent (harbor | toy)."""

    gateway: str
    api_key: str
    base_model: str
    rank: int = 16
    tasks_dir: str = ""
    tasks_per_step: int = 4
    samples_per_task: int = 4
    steps: int = 10
    concurrency: int = 16
    max_seq_len: int = 65536
    loss_fn: str = "ppo"
    learning_rate: float = 3e-5
    agent: str = "harbor"


def load_agent(name: str) -> Callable[..., Any]:
    """Return the agent function: harbor_agent_function.run (examples/experimental/harbor) or toy_tool_agent.run."""
    raise NotImplementedError


def list_tasks(tasks_dir: str) -> list[str]:
    """Task ids = the Harbor task directories under tasks_dir (a Terminal-Bench-2 checkout works as-is)."""
    raise NotImplementedError


async def run_trial(
    session_client, agent_run, task_id: str, model_path: str, args: LoopArgs
) -> tuple[dict, float, str]:
    """Bind a fresh session to model_path, run the agent against it, fetch the trajectory, delete the session, return (trajectory, reward, exit_status)."""
    raise NotImplementedError


async def train_step(
    training_client, session_client, agent_run, tasks: list[str], model_path: str, args: LoopArgs
) -> str:
    """Run tasks × samples_per_task trials under a semaphore, build GRPO Datums per task group, forward_backward + optim_step, save_weights_for_sampler, return the new sampler path."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Parse LoopArgs, create the LoRA training client, save sampler version 0, loop train_step for args.steps."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
