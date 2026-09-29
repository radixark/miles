"""NCCL options for the trainer's weight-transfer communicator."""

import os
from collections.abc import Mapping

import torch.distributed as dist

CHANNEL_ENV_KEYS = (
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_NRINGS",
    "NCCL_MAX_NRINGS",
)


def validate_channel_env(env: Mapping[str, str], channels: int, *, role: str) -> None:
    for key in CHANNEL_ENV_KEYS:
        value = env.get(key)
        if value is not None and value != str(channels):
            raise ValueError(
                f"{role} sets {key}={value}, but deterministic serving requires {key}={channels} "
                "for NCCL weight transfer. Remove the conflicting override or set it to the required value "
                "before launching workers. To change the shared count, set SGLANG_DETERMINISTIC_NCCL_NCHANNELS "
                "in the job environment."
            )


def weight_update_nccl_options(channels: int | None):
    if channels is None:
        return None
    validate_channel_env(os.environ, channels, role="trainer")
    process_group = getattr(dist, "ProcessGroupNCCL", None)
    if process_group is None:
        raise RuntimeError("Deterministic NCCL weight transfer requires a PyTorch build with NCCL support.")
    options = process_group.Options()
    config = getattr(options, "config", None)
    if config is None or not all(hasattr(config, name) for name in ("min_ctas", "max_ctas")):
        raise RuntimeError(
            "Deterministic NCCL weight transfer requires PyTorch/NCCL communicator min_ctas and max_ctas options. "
            "Use a supported Miles CUDA image."
        )
    config.min_ctas = channels
    config.max_ctas = channels
    return options
