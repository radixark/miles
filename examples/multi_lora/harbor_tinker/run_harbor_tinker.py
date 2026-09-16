"""Harbor × Tinker RL on the multi-LoRA gateway: cookbook's training loop with our Harbor plug-ins.

Skeleton: functions document what they will do; bodies land in follow-up commits.

The loop is ``tinker_cookbook.rl.train.main`` unchanged. This file only builds its ``Config``:
``base_url`` = the gateway, ``model_name`` = the frozen base, ``dataset_builder`` = ``HarborDatasetBuilder``,
``rollout_error_tolerance`` = ``SessionRolloutStrategy`` (the cookbook accepts a strategy instance there).

Usage (stage 1, internal AgentENV sandbox):
    HARBOR_ENV_TYPE=e2b HARBOR_TASKS_DIR=<terminal-bench-2 checkout> TINKER_API_KEY=tml-... \\
    python run_harbor_tinker.py --gateway http://<gateway>:10613 --base-model Qwen3-30B-A3B

Stage 0 swaps the Harbor harness for ``toy_tool_agent.run`` with ``--agent toy``.
"""

from __future__ import annotations

from tinker_cookbook.rl import train


def build_config(
    gateway: str,
    base_model: str,
    tasks_dir: str,
    log_path: str,
    *,
    agent_name: str = "terminus-2",
    lora_rank: int = 16,
    groups_per_batch: int = 4,
    group_size: int = 4,
    concurrency: int = 16,
    max_seq_len: int = 65536,
    loss_fn: str = "ppo",
    learning_rate: float = 3e-5,
) -> train.Config:
    """Cookbook Config wired to the gateway: HarborDatasetBuilder as dataset_builder, SessionRolloutStrategy as rollout_error_tolerance, ppo loss; everything else cookbook defaults."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Parse the CLI into build_config(...) and run tinker_cookbook.rl.train.main(config)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
