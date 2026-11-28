"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own state-dict adapter.

A fused projection (q/k/v, gate/up) carries one ``lora_a`` for all the HF projections it holds and stacks their
``lora_b`` rows exactly like its base weight, so ``lora_b`` splits through the adapter the way the base weight does
and each HF projection gets a copy of the shared ``lora_A``.

Routed experts (3D ``lora_a``) are shared-outer, the layout of megatron's ``GroupedExpertsSpec`` and SGLang's
``--experts-shared-outer-loras``: gate/up export as ``experts.w1`` / ``experts.w3`` with one ``lora_A`` [1, r, D]
each and per-expert ``lora_B`` [E, F, r]; down exports as ``experts.w2`` with per-expert ``lora_A`` [E, r, F] and one
``lora_B`` [1, D, r]. Their HF targets are the packed ``experts.gate_up_proj`` / ``experts.down_proj``.

Needs only torch, so the weight iterator imports without torchtitan."""

from collections.abc import Iterator, Mapping

import torch

_PEFT_FACTOR_BY_TITAN_NAME = {"lora_a": "lora_A", "lora_b": "lora_B"}
# the leaves megatron's GroupedExpertsSpec exports shared-outer experts under
_GATE_UP_LEAVES = ("w1", "w3")
_DOWN_LEAF = "w2"
# routed-expert adapters run grouped matmuls, whose operands need 16-byte aligned strides: rank * sizeof(bf16)
GROUPED_LORA_RANK_MULTIPLE = 8


def iter_hf_adapter_tensors(sd_adapter, model_state: Mapping[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """The adapter params in ``model_state`` under HF PEFT names (``...q_proj.lora_A.weight``)."""
    for key, lora_b in model_state.items():
        parts = split_adapter_key(key)
        if parts is None or parts[1] != "lora_b":
            continue
        projection = parts[0]
        lora_a = model_state[f"{projection}.lora_a.weight"]
        if lora_a.ndim == 3:
            yield from _shared_outer_expert_tensors(_experts_module(sd_adapter, projection, lora_b), lora_a, lora_b)
            continue
        for hf_name, hf_lora_b in sd_adapter.to_hf({f"{projection}.weight": lora_b}).items():
            module = hf_name.removesuffix(".weight")
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_a']}.weight", lora_a
            yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_b']}.weight", hf_lora_b


def _shared_outer_expert_tensors(
    experts: str, lora_a: torch.Tensor, lora_b: torch.Tensor
) -> Iterator[tuple[str, torch.Tensor]]:
    if lora_b.ndim == 4:
        # gate/up: lora_a [2, r, D] shared, lora_b [E, 2, F, r] per expert
        for index, leaf in enumerate(_GATE_UP_LEAVES):
            yield f"{experts}.{leaf}.lora_A.weight", lora_a[index : index + 1]
            yield f"{experts}.{leaf}.lora_B.weight", lora_b[:, index]
        return
    # down: lora_a [E, r, F] per expert, lora_b [1, D, r] shared
    yield f"{experts}.{_DOWN_LEAF}.lora_A.weight", lora_a
    yield f"{experts}.{_DOWN_LEAF}.lora_B.weight", lora_b


def _experts_module(sd_adapter, projection: str, lora_b: torch.Tensor) -> str:
    """The HF module holding a routed-expert projection's stacked tensor (``...mlp.experts``)."""
    (name,) = sd_adapter.to_hf({f"{projection}.weight": torch.empty(lora_b.shape, device="meta")})
    return name.rsplit(".", 1)[0]


def is_adapter_key(key: str) -> bool:
    return split_adapter_key(key) is not None


def split_adapter_key(key: str) -> tuple[str, str] | None:
    """``(base projection FQN, factor)`` for a titan adapter param key, else None."""
    for factor in _PEFT_FACTOR_BY_TITAN_NAME:
        suffix = f".{factor}.weight"
        if key.endswith(suffix):
            return key.removesuffix(suffix), factor
    return None


def hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The HF modules a titan projection's weight exports as; several for a fused projection."""
    probe = torch.empty(weight_shape, device="meta")
    return [name.removesuffix(".weight") for name in sd_adapter.to_hf({f"{projection}.weight": probe})]


def hf_expert_target(stacked_hf_name: str, *, num_linears: int) -> str:
    """The packed HF target of a routed-expert projection the checkpoint stores as one stacked tensor."""
    experts = stacked_hf_name.rsplit(".", 1)[0]
    return f"{experts}.gate_up_proj" if num_linears > 1 else f"{experts}.down_proj"


def merged_weight(base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, *, scale: float) -> torch.Tensor:
    """``base + scale * lora_b @ lora_a`` in the base's titan layout and dtype. The batched matmul broadcasts every
    layout this backend trains: [O, r] @ [r, I], stacked [n, O, r] @ [r, I], and the shared-outer experts
    [E, 2, F, r] @ [2, r, D] and [1, D, r] @ [E, r, F]."""
    merged = lora_b.float() @ lora_a.float()
    return merged.mul_(scale).add_(base.float()).to(base.dtype)
