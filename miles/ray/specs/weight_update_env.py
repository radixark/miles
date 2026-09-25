"""Resolve channel limits before NCCL weight-transfer workers start."""

import logging
import os
from collections.abc import Mapping
from functools import partial

logger = logging.getLogger(__name__)
_CHANNEL_KEYS = ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")


def resolve_weight_update_env(args, config, *, is_hip: bool) -> dict[str, dict[str, str]]:
    """Return per-pool environments for managed CUDA broadcast participants."""
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
            participants.append(f"inference-engine-{model_idx}-{group_idx}")
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
    channel_env = dict.fromkeys(_CHANNEL_KEYS, str(channels))
    # Trainer-specific values override the inherited job environment, as in train.py.
    trainer_env = {**os.environ, **getattr(args, "train_env_vars", {})}
    _validate_channels(trainer_env, channel_env, "trainer")
    _validate_channels(os.environ, channel_env, "serving")
    logger.info(
        "NCCL weight-transfer workers will use %s channels; trainer collective performance may change", channels
    )
    return {
        "trainer-actor": channel_env,
        **{name: {**channel_env, "SGLANG_DETERMINISTIC_NCCL_NCHANNELS": str(channels)} for name in participants},
    }


def _deterministic_channels() -> int | None:
    # Keep SGLang optional for launches that do not need its deterministic policy.
    from sglang.srt.environ import envs

    setting = getattr(envs, "SGLANG_DETERMINISTIC_NCCL_NCHANNELS", None)
    return setting.get() if setting is not None else None


def _validate_channels(actual: Mapping[str, str], required: Mapping[str, str], role: str) -> None:
    for key, expected in required.items():
        value = actual.get(key)
        if value is not None and value != expected:
            raise ValueError(
                f"{role} sets {key}={value}, but deterministic serving requires {key}={expected} "
                "for NCCL weight transfer. Remove the conflicting override or set it to the required value "
                "before launching workers. To change the shared count, set SGLANG_DETERMINISTIC_NCCL_NCHANNELS "
                "in the job environment."
            )


def _worker_env(ctx, *, original, required: dict[str, str], role: str) -> dict[str, str]:
    env = original(ctx)
    _validate_channels(env, required, role)
    return {**env, **required}


def apply_weight_update_env(specs, environments: dict[str, dict[str, str]]):
    """Keep the resolved policy in each launch spec, including restarted workers."""
    updated = []
    for spec in specs:
        if required := environments.get(spec.name):
            env_var = partial(_worker_env, original=spec.env_var, required=required, role=spec.name)
            spec = spec.model_copy(update={"env_var": env_var})
        updated.append(spec)
    return updated
