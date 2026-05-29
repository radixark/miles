from __future__ import annotations

import os

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def enable_sglang_batch_invariant_mode() -> None:
    """Enable deterministic runtime knobs expected by the SGLang backend."""

    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"


def ensure_batch_invariant_mode_from_config(config: TransformerConfig) -> None:
    if not getattr(config, "batch_invariant_mode", False):
        return

    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()
