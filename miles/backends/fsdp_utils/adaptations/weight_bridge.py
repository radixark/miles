"""WeightBridge: the registered train->rollout parameter-name/shape contract for the FSDP backend.

HF/FSDP training and the SGLang rollout loader don't always agree on param names/shapes (e.g.
transformers>=5.6 stores qwen3_moe experts as one batched tensor; SGLang wants per-expert names). Each
disagreeing model_type registers a ParamTransform (a ``matches`` selector + an ``expand`` that rewrites
the materialized tensor) instead of editing the sync loop -- the FSDP analogue of Megatron's
megatron_to_hf. ``expand`` never touches DTensor/device state, so transforms are CPU-unit-testable.
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
from transformers.core_model_loading import revert_weight_conversion


class ParamTransform(NamedTuple):
    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor, torch.nn.Module], Iterable[tuple[str, torch.Tensor]]]


# model_type -> registered transforms, tried in registration order
_REGISTRY: dict[str, list[ParamTransform]] = {}


def register_param_transform(model_type: str, matches: Callable, expand: Callable) -> None:
    _REGISTRY.setdefault(model_type, []).append(ParamTransform(matches, expand))


def get_param_transform(name: str, param, model_type: str):
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for transform in _REGISTRY.get(model_type, ()):
        if transform.matches(name, param):
            return transform.expand
    return None


# batched-expert archs (qwen3_moe, glm4_moe_lite, ...): unfuse into the per-expert names SGLang expects.
def _batched_experts_matches(name: str, param) -> bool:
    return getattr(param, "dim", lambda: 0)() == 3 and (
        name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    )


def _hf_unfuse_experts_expand(name: str, full: torch.Tensor, model) -> Iterable[tuple[str, torch.Tensor]]:
    """Unfuse a batched experts tensor through transformers' own save path (``revert_weight_conversion``),
    yielding exactly what ``save_pretrained`` would write -- the on-disk dialect SGLang's loader expects.
    The revert ops slice views, so contiguity is re-asserted before streaming."""
    for out_name, tensor in revert_weight_conversion(model, {name: full}).items():
        yield out_name, tensor.contiguous()
