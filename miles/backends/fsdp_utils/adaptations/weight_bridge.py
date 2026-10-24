"""Train->rollout parameter rewrites shared by the FSDP ``ArchAdapter``s.

HF/FSDP training and the SGLang rollout loader don't always agree on param names/shapes (e.g.
transformers>=5.6 stores qwen3_moe experts as one batched tensor; SGLang wants per-expert names). An
adapter's ``param_transform`` picks one of these rewrites per param -- the FSDP analogue of Megatron's
megatron_to_hf. The rewrites never touch DTensor/device state, so they are CPU-unit-testable.
"""

from collections.abc import Iterable

import torch
from transformers.core_model_loading import revert_weight_conversion


def is_batched_experts_param(name: str, param: torch.Tensor) -> bool:
    """True for the 3D batched-experts tensors transformers>=5.6 fuses (qwen3_moe, glm4_moe_lite, ...)."""
    return param.dim() == 3 and (name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"))


def unfuse_batched_experts(
    name: str, full: torch.Tensor, model: torch.nn.Module
) -> Iterable[tuple[str, torch.Tensor]]:
    """Unfuse a batched experts tensor through transformers' own save path (``revert_weight_conversion``),
    yielding exactly what ``save_pretrained`` would write -- the on-disk dialect SGLang's loader expects.
    The revert ops slice views, so contiguity is re-asserted before streaming."""
    for out_name, tensor in revert_weight_conversion(model, {name: full}).items():
        yield out_name, tensor.contiguous()
