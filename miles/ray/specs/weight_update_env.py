"""Resolve channel limits before NCCL weight-transfer workers start."""

import copy
import logging
import os
from functools import partial

from miles.backends.training_utils.weight_update.nccl import validate_channel_env

logger = logging.getLogger(__name__)


def resolve_weight_update_channels(
    args, config, *, is_hip: bool, trainer_pool_ids: list[str], engine_pool_ids: dict[tuple[int, int], str]
) -> dict[str, int]:
    """Return the shared channel count for each managed broadcast participant."""
    if (
        is_hip
        or getattr(args, "colocate", False)
        or getattr(args, "debug_train_only", False)
        or getattr(args, "debug_rollout_only", False)
        or getattr(args, "debug_skip_weight_update", False)
        or getattr(args, "rollout_external", False)
        or getattr(args, "update_weight_transfer_mode", "broadcast") != "broadcast"
    ):
        return {}

    participants = []
    deterministic = False
    for model_idx, model in enumerate(config.models):
        if not model.update_weights:
            continue
        for group_idx, group in enumerate(model.server_groups):
            if group.worker_type == "placeholder":
                continue
            device = group.overrides.get("device", getattr(args, "sglang_device", None))
            if device not in (None, "cuda"):
                continue
            participants.append(engine_pool_ids[model_idx, group_idx])
            # SGLang also enables determinism for prefill-only and on-policy contracts.
            enabled = any(
                group.overrides.get(name, getattr(args, f"sglang_{name}", None))
                for name in (
                    "enable_deterministic_inference",
                    "enable_prefill_only_deterministic_inference",
                    "rl_on_policy_target",
                    "true_on_policy_contract",
                )
            )
            tp_size = group.overrides.get("tp_size", group.num_gpus_per_engine)
            deterministic |= bool(enabled and tp_size > 1)
    if not deterministic:
        return {}

    channels = _deterministic_channels()
    if channels is None:
        return {}
    if channels <= 0:
        raise ValueError("SGLANG_DETERMINISTIC_NCCL_NCHANNELS must be a positive integer")
    trainer_env = {**os.environ, **getattr(args, "train_env_vars", {})}
    validate_channel_env(trainer_env, channels, role="trainer")
    validate_channel_env(os.environ, channels, role="serving")
    logger.info("NCCL weight-transfer groups will use %s channels", channels)
    return dict.fromkeys([*trainer_pool_ids, *participants], channels)


def _deterministic_channels() -> int | None:
    # Keep SGLang optional for launches that do not need its deterministic policy.
    from sglang.srt.environ import envs

    setting = getattr(envs, "SGLANG_DETERMINISTIC_NCCL_NCHANNELS", None)
    return setting.get() if setting is not None else None


def _worker_env(ctx, *, original, channels: int, role: str) -> dict[str, str]:
    env = original(ctx)
    validate_channel_env(env, channels, role=role)
    if role.startswith("trainer-engine-"):
        return env
    return {
        **env,
        "NCCL_MIN_NCHANNELS": str(channels),
        "NCCL_MAX_NCHANNELS": str(channels),
        "SGLANG_DETERMINISTIC_NCCL_NCHANNELS": str(channels),
    }


def _trainer_kwargs(ctx, *, original, channels: int):
    kwargs = original(ctx)
    args = copy.copy(kwargs["args"])
    args._weight_update_nccl_channels = channels
    return {**kwargs, "args": args}


def apply_weight_update_channels(specs, channels_by_pool: dict[str, int]):
    """Preserve the transfer policy through worker creation and restart."""
    updated = []
    for spec in specs:
        if (channels := channels_by_pool.get(spec.name)) is not None:
            updates = {"env_var": partial(_worker_env, original=spec.env_var, channels=channels, role=spec.name)}
            if spec.name.startswith("trainer-engine-"):
                updates["ctor_kwargs"] = partial(_trainer_kwargs, original=spec.ctor_kwargs, channels=channels)
            spec = spec.model_copy(update=updates)
        updated.append(spec)
    return updated
