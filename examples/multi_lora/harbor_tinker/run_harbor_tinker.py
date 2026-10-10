import asyncio
import os

import chz
from examples.multi_lora.harbor_tinker.harbor_env import HarborDatasetBuilder, SessionRolloutStrategy
from tinker_cookbook.rl import train


@chz.chz
class HarborTinkerConfig:
    gateway: str
    model_name: str
    renderer_name: str
    tasks_dir: str
    api_key: str | None = None
    log_path: str = "/tmp/harbor-tinker"
    agent_name: str = "terminus-2"
    lora_rank: int = 16
    groups_per_batch: int = 4
    group_size: int = 4
    epochs: int = 1
    advertised_host: str = "127.0.0.1"
    listen_host: str = "127.0.0.1"
    max_parallel_trials_per_group: int = 4
    max_datum_tokens: int = 32768
    max_turns: int = 512
    max_tokens: int = 8192
    temperature: float = 1.0
    loss_fn: str = "ppo"
    learning_rate: float = 3e-5
    max_steps: int | None = None
    save_every: int = 5
    wandb_project: str | None = None
    wandb_name: str | None = None


def build_config(config: HarborTinkerConfig) -> train.Config:
    return train.Config(
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        base_url=config.gateway,
        recipe_name="harbor-tinker",
        log_path=config.log_path,
        dataset_builder=HarborDatasetBuilder(
            tasks_dir=config.tasks_dir, groups_per_batch=config.groups_per_batch,
            group_size=config.group_size, agent_name=config.agent_name, epochs=config.epochs,
        ),
        rollout_error_tolerance=SessionRolloutStrategy(
            renderer_name=config.renderer_name, advertised_host=config.advertised_host,
            listen_host=config.listen_host, max_parallel_trials_per_group=config.max_parallel_trials_per_group,
            max_datum_tokens=config.max_datum_tokens, max_turns=config.max_turns,
        ),
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        loss_fn=config.loss_fn,
        max_steps=config.max_steps,
        save_every=config.save_every,
        ttl_seconds=None,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )


def main(config: HarborTinkerConfig) -> None:
    if config.api_key is not None:
        os.environ["TINKER_API_KEY"] = config.api_key
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("set TINKER_API_KEY or pass api_key")
    if not os.environ.get("HARBOR_ENV_TYPE"):
        raise ValueError("set HARBOR_ENV_TYPE to the sandbox provider")
    os.environ["HARBOR_TASKS_DIR"] = config.tasks_dir
    asyncio.run(train.main(build_config(config)))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
