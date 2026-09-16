"""Harbor × Tinker RL on the multi-LoRA gateway: cookbook's training loop with our Harbor plug-ins.

Skeleton: functions document what they will do; bodies land in follow-up commits.

Reused, not reimplemented: ``tinker_cookbook.rl.train.Config`` / ``train.main`` (rollouts, advantages, Datums,
forward_backward / optim_step, checkpoints, metrics, wandb via ml_log), ``chz.nested_entrypoint`` for the CLI (the
same wiring as ``tinker_cookbook/recipes/rl_loop.py``), and ``miles.rollout.agentic.credentials`` (``preflight_sdk``,
``PROVIDER_CREDENTIALS``, ``credential_available``) for the sandbox preflight. This file only builds the Config.

Usage (stage 1, internal AgentENV sandbox):
    HARBOR_ENV_TYPE=e2b HARBOR_TASKS_DIR=<terminal-bench-2 checkout> TINKER_API_KEY=tml-... \\
    python run_harbor_tinker.py gateway=http://<gateway>:10613 model_name=Qwen/Qwen3-30B-A3B

Stage 0 needs no extra agent: Harbor's golden agent (``agent_name=oracle``, no model) proves the sandbox
round trip, and one terminus-2 trial on an easy task (``fix-git``) exercises the collector's recording.
"""

from __future__ import annotations

import chz
from tinker_cookbook.rl import train


@chz.chz
class HarborTinkerConfig:
    """CLI knobs on top of cookbook's train.Config: gateway URL, tasks dir, harness, batch shape, concurrency, max_seq_len."""

    gateway: str
    model_name: str
    tasks_dir: str
    log_path: str = "/tmp/harbor-tinker"
    agent_name: str = "terminus-2"
    lora_rank: int = 16
    groups_per_batch: int = 4
    group_size: int = 4
    concurrency: int = 16
    max_seq_len: int = 65536
    max_tokens: int = 8192
    loss_fn: str = "ppo"
    learning_rate: float = 3e-5
    max_steps: int | None = None


def preflight_sandbox() -> None:
    """Fail fast before training: HARBOR_ENV_TYPE and HARBOR_TASKS_DIR set, provider key file present (credential_available), e2b SDK ≥ 2.12 (preflight_sdk), E2B_API_URL/sandboxes answers with the key (401 = bad key, refused = cluster stopped or off the tailnet)."""
    raise NotImplementedError


def build_config(config: HarborTinkerConfig) -> train.Config:
    """train.Config with base_url=gateway, dataset_builder=HarborDatasetBuilder, rollout_error_tolerance=SessionRolloutStrategy, loss_fn/lora_rank/learning_rate/max_tokens from the CLI; everything else cookbook defaults."""
    raise NotImplementedError


def main(config: HarborTinkerConfig) -> None:
    """preflight_sandbox, then asyncio.run(train.main(build_config(config)))."""
    raise NotImplementedError


if __name__ == "__main__":
    chz.nested_entrypoint(main)
