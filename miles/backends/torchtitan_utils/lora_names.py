"""HF PEFT names for the torchtitan backend's LoRA adapter params, through the model's own state-dict adapter.

A fused projection (q/k/v, gate/up) carries one ``lora_a`` for all the HF projections it holds and stacks their
``lora_b`` rows exactly like its base weight, so ``lora_b`` splits through the adapter the way the base weight does
and each HF projection gets a copy of the shared ``lora_A``. A canonical adapter (``lora_a.{i}`` / ``lora_b.{i}``) gives
each HF projection of a fused weight its own A; its B rows pack into the fused output like the base weight's. Needs
only torch, so the weight iterator imports without torchtitan."""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

_PEFT_FACTOR_BY_TITAN_NAME = {"lora_a": "lora_A", "lora_b": "lora_B"}
_ADAPTER_KEY = re.compile(r"(?P<projection>.+)\.(?P<factor>lora_a|lora_b)(?:\.(?P<component>\d+))?\.weight")


@dataclass(frozen=True)
class ComponentPacking:
    """How a canonical adapter's per-component outputs pack into its fused projection's output.

    ``stacked``: a ``[num_linears, F, D]`` weight, component i is linear i. ``qkv``: ``QKVLinear.wqkv``, whose rows are
    ``(kv groups, heads_per_kv + 2, head_dim)`` with components q, k, v; packing any number of whole kv groups, so it
    holds for a TP shard as for the whole weight.
    """

    kind: Literal["stacked", "qkv"]
    num_components: int
    heads_per_kv: int = 0
    head_dim: int = 0

    def pack(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-component outputs (last dim: the component's rows) as the fused output's last dim."""
        assert len(outputs) == self.num_components, f"{len(outputs)} outputs for {self.num_components} components"
        if self.kind == "stacked":
            return torch.cat(outputs, dim=-1)
        q, k, v = outputs
        return torch.cat(
            [
                q.unflatten(-1, (-1, self.heads_per_kv, self.head_dim)),
                k.unflatten(-1, (-1, 1, self.head_dim)),
                v.unflatten(-1, (-1, 1, self.head_dim)),
            ],
            dim=-2,
        ).flatten(-3)

    def pack_lora_b(self, lora_bs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-component ``[rows, rank]`` B as one B shaped like the fused base weight."""
        if self.kind == "stacked":
            return torch.stack(list(lora_bs))
        return self.pack([b.t() for b in lora_bs]).t()


def iter_hf_adapter_tensors(
    sd_adapter, model_state: Mapping[str, torch.Tensor], packings: Mapping[str, ComponentPacking] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """The adapter params in ``model_state`` under HF PEFT names (``...q_proj.lora_A.weight``).

    ``packings``: by projection FQN, for the canonical adapters in ``model_state``.
    """
    for key, lora_b in model_state.items():
        parts = split_adapter_key(key)
        if parts is None or parts[1] != "lora_b":
            continue
        projection, _, component = parts
        if component is None:
            lora_a = model_state[f"{projection}.lora_a.weight"]
            for hf_name, hf_lora_b in sd_adapter.to_hf({f"{projection}.weight": lora_b}).items():
                yield from _peft_pair(hf_name, lora_a, hf_lora_b)
        elif component == 0:
            yield from _iter_canonical(sd_adapter, model_state, projection, packings[projection])


def _iter_canonical(
    sd_adapter, model_state: Mapping[str, torch.Tensor], projection: str, packing: ComponentPacking
) -> Iterator[tuple[str, torch.Tensor]]:
    lora_bs = [model_state[f"{projection}.lora_b.{i}.weight"] for i in range(packing.num_components)]
    hf_lora_bs = sd_adapter.to_hf({f"{projection}.weight": packing.pack_lora_b(lora_bs)})
    # the model's own split names each component: rows packed from component i come out under its HF name
    labels = packing.pack_lora_b([torch.full((b.shape[0], 1), i) for i, b in enumerate(lora_bs)])
    for hf_name, label in sd_adapter.to_hf({f"{projection}.weight": labels}).items():
        component = int(label.flatten()[0])
        assert (label == component).all(), f"{hf_name} mixes rows of several components of {projection}"
        yield from _peft_pair(hf_name, model_state[f"{projection}.lora_a.{component}.weight"], hf_lora_bs[hf_name])


def _peft_pair(hf_name: str, lora_a: torch.Tensor, lora_b: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    module = hf_name.removesuffix(".weight")
    yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_a']}.weight", lora_a
    yield f"{module}.{_PEFT_FACTOR_BY_TITAN_NAME['lora_b']}.weight", lora_b


def is_adapter_key(key: str) -> bool:
    return split_adapter_key(key) is not None


def split_adapter_key(key: str) -> tuple[str, str, int | None] | None:
    """``(base projection FQN, factor, canonical component or None)`` for a titan adapter param key, else None."""
    match = _ADAPTER_KEY.fullmatch(key)
    if match is None:
        return None
    component = match["component"]
    return match["projection"], match["factor"], None if component is None else int(component)


def hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The HF modules a titan projection's weight exports as; several for a fused projection."""
    probe = torch.empty(weight_shape, device="meta")
    return [name.removesuffix(".weight") for name in sd_adapter.to_hf({f"{projection}.weight": probe})]
