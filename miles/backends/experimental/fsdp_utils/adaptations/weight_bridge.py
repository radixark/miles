"""WeightBridge: the registered train->rollout parameter-name/shape contract for the FSDP backend.

HF/FSDP training and the SGLang rollout loader don't always agree on param names/shapes (e.g.
transformers>=5.6 stores qwen3_moe experts as one batched tensor; SGLang wants per-expert names). Each
disagreeing model_type registers a ParamTransform (a ``matches`` selector + a pure ``expand`` that
rewrites the materialized tensor) instead of editing the sync loop -- the FSDP analogue of Megatron's
megatron_to_hf. ``expand`` stays pure (no DTensor/device) so transforms are CPU-unit-testable.
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch


class ParamTransform(NamedTuple):
    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]]


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


# qwen3_moe: split transformers>=5.6 batched experts into the per-expert names SGLang expects.
def _qwen3_moe_matches(name: str, param) -> bool:
    return getattr(param, "dim", lambda: 0)() == 3 and (
        name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    )


def _qwen3_moe_expand(name: str, full: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
    """experts.gate_up_proj [E,2I,H] -> experts.{i}.{gate,up}_proj.weight; down_proj [E,H,I] -> per-expert."""
    prefix = name.rsplit(".", 1)[0]  # ...mlp.experts
    num_experts = full.shape[0]
    if name.endswith(".gate_up_proj"):
        half = full.shape[1] // 2  # fused rows are [gate | up]
        for i in range(num_experts):
            yield f"{prefix}.{i}.gate_proj.weight", full[i, :half, :].contiguous()
            yield f"{prefix}.{i}.up_proj.weight", full[i, half:, :].contiguous()
    else:  # .down_proj
        for i in range(num_experts):
            yield f"{prefix}.{i}.down_proj.weight", full[i].contiguous()


# archs with this batched-expert layout (qwen3_moe, glm4_moe_lite, ...) register these in their spec.
