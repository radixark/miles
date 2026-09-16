"""Harbor × Tinker RL on the multi-LoRA gateway: HarborTinkerConfig → cookbook train.Config (HarborDatasetBuilder + SessionRolloutStrategy) → train.main, with a sandbox preflight; usage and run order in README.md."""

from __future__ import annotations

import asyncio
import os

import chz
import httpx
from examples.multi_lora.harbor_tinker.harbor_env import HarborDatasetBuilder, SessionRolloutStrategy
from tinker_cookbook.rl import train

from miles.rollout.agentic.credentials import (
    PROVIDER_CREDENTIALS,
    credential_available,
    preflight_sdk,
    resolve_provider_api_key,
)

RECIPE_NAME = "harbor-tinker"


@chz.chz
class HarborTinkerConfig:
    """CLI knobs on top of cookbook's train.Config: gateway URL, tasks dir, harness, batch shape, concurrency, lengths."""

    gateway: str
    model_name: str
    tasks_dir: str
    api_key: str | None = None  # the gateway tenant key; TINKER_API_KEY when unset
    log_path: str = "/tmp/harbor-tinker"
    agent_name: str = "terminus-2"
    lora_rank: int = 16
    groups_per_batch: int = 4
    group_size: int = 4
    epochs: int = 1  # passes over the task list; steps = ceil(tasks * epochs / groups_per_batch), capped by max_steps
    concurrency: int = 16
    max_seq_len: int = 65536
    max_tokens: int = 8192
    max_datum_tokens: int = (
        32768  # set to the gateway's per-datum cap (min of model max_position_embeddings and --max-tokens-per-gpu); longer turns are dropped client-side
    )
    temperature: float = 1.0
    loss_fn: str = "ppo"
    learning_rate: float = 3e-5
    max_steps: int | None = None
    save_every: int = 5
    wandb_project: str | None = None
    record_path: str | None = None  # per-trajectory JSONL (task, turns, token counts, reward) for experiment notes


def preflight_sandbox() -> None:
    """Fail fast before training: HARBOR_ENV_TYPE and HARBOR_TASKS_DIR set, a known provider with its credential in place (credential_available), its SDK importable at the floor version (preflight_sdk), and for a self-hosted AgentENV (E2B_API_URL) the endpoint answering the key (401 = bad key, refused = cluster stopped or off the tailnet)."""
    env_type = os.environ.get("HARBOR_ENV_TYPE", "").strip().lower()
    if not env_type:
        raise RuntimeError("set HARBOR_ENV_TYPE to the sandbox provider Harbor runs trials on (e2b for AgentENV)")
    if not os.environ.get("HARBOR_TASKS_DIR", "").strip():
        raise RuntimeError("set HARBOR_TASKS_DIR to the directory holding one Harbor task dir per task id (tasks_dir)")
    spec = PROVIDER_CREDENTIALS.get(env_type)
    if spec is None:
        raise RuntimeError(
            f"HARBOR_ENV_TYPE={env_type!r} has no credential spec; known providers: {sorted(PROVIDER_CREDENTIALS)}"
        )
    key_path = os.environ.get(spec["file_env_var"], "").strip()
    if not credential_available(spec, arg_path=key_path):
        raise RuntimeError(
            f"{spec['provider']} credential missing: put it in {key_path or spec['default_path']} or set "
            f"{' + '.join(spec['key_env_vars'])}. Provision with: {spec['provision_hint']}"
        )
    preflight_sdk(spec["sdk"], spec["sdk_hint"], spec.get("sdk_min_version"))
    api_url = os.environ.get("E2B_API_URL", "").strip()
    if env_type == "e2b" and api_url:
        api_key = resolve_provider_api_key(spec["key_env_vars"][0], spec["file_env_var"], spec["default_path"])
        _probe_agentenv(api_url, api_key)


def _probe_agentenv(api_url: str, api_key: str) -> None:
    """GET {E2B_API_URL}/sandboxes with the key: a refused connection means the cluster is stopped or this host is off its tailnet, 401 means the key is wrong; anything else is left to Harbor's own errors."""
    try:
        response = httpx.get(f"{api_url.rstrip('/')}/sandboxes", headers={"X-API-Key": api_key}, timeout=10.0)
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"E2B_API_URL={api_url} is unreachable ({error}): is the AgentENV cluster running and this host on its tailnet?"
        ) from error
    if response.status_code == 401:
        raise RuntimeError(f"E2B_API_URL={api_url} rejected the key (401): check the E2B_API_KEY file")


def build_config(config: HarborTinkerConfig) -> train.Config:
    """train.Config(model_name, base_url=gateway, recipe_name, log_path, dataset_builder=HarborDatasetBuilder(...), rollout_error_tolerance=SessionRolloutStrategy(...), learning_rate, lora_rank, max_tokens, temperature, loss_fn, max_steps, save_every, wandb_project); the cookbook's ttl_seconds default is accepted by the gateway."""
    api_key = config.api_key or os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "no gateway tenant key: pass api_key=... or set TINKER_API_KEY (the key the Tinker SDK uses)"
        )
    return train.Config(
        model_name=config.model_name,
        base_url=config.gateway,
        recipe_name=RECIPE_NAME,
        log_path=config.log_path,
        dataset_builder=HarborDatasetBuilder(
            tasks_dir=config.tasks_dir,
            groups_per_batch=config.groups_per_batch,
            group_size=config.group_size,
            agent_name=config.agent_name,
            epochs=config.epochs,
        ),
        rollout_error_tolerance=SessionRolloutStrategy(
            gateway_url=config.gateway,
            api_key=api_key,
            concurrency=config.concurrency,
            max_seq_len=config.max_seq_len,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_datum_tokens=config.max_datum_tokens,
            record_path=config.record_path,
        ),
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        loss_fn=config.loss_fn,
        max_steps=config.max_steps,
        save_every=config.save_every,
        wandb_project=config.wandb_project,
    )


def main(config: HarborTinkerConfig) -> None:
    """Export what the trial runner and the SDK read (HARBOR_TASKS_DIR = tasks_dir, TINKER_API_KEY), preflight the sandbox, then run cookbook train.main."""
    os.environ.setdefault("HARBOR_TASKS_DIR", config.tasks_dir)
    if config.api_key:
        os.environ.setdefault("TINKER_API_KEY", config.api_key)
    preflight_sandbox()
    asyncio.run(train.main(build_config(config)))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
