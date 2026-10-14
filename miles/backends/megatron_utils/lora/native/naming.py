"""HF module names of native adapters, from the owner of each architecture's Megatron->HF naming."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import torch
import torch.nn as nn

_EXPERT_INDEX_PATTERN = re.compile(r"\.experts\.\d+\.")


class HFNaming(Protocol):
    def block_prefix(self, layer: nn.Module, host_weight: nn.Parameter, leaves: Sequence[str]) -> str:
        """HF prefix of the block whose projections ``leaves`` the host weight carries, e.g. ``model.layers.3.self_attn.``."""
        ...


class ConverterNaming:
    """Names adapters after the HF weights the raw-mode converter makes of their hosts.

    That converter names every base weight the run syncs and exports, so an adapter and its
    host cannot disagree. The converter is fed a meta tensor of the host's global shape:
    names depend on shapes, never on values.
    """

    def __init__(self, args, model_name: str, tp_size: int, is_hf_name: Callable[[str], bool]):
        self.args = args
        self.model_name = model_name
        self.tp_size = tp_size
        self.is_hf_name = is_hf_name

    def block_prefix(self, layer: nn.Module, host_weight: nn.Parameter, leaves: Sequence[str]) -> str:
        from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf_names

        name_in_layer = next(name for name, param in layer.named_parameters() if param is host_weight)
        global_name = f"module.module.decoder.layers.{layer.layer_number - 1}.{name_in_layer}"
        meta_weight = torch.empty(self._global_shape(host_weight), dtype=host_weight.dtype, device="meta")
        hf_weight_names = convert_to_hf_names(self.args, self.model_name, global_name, meta_weight)
        # target selectors match HF names; a converter name outside that namespace is never selected
        names_outside_hf_namespace = [name for name in hf_weight_names if not self.is_hf_name(name)]
        assert not names_outside_hf_namespace, (
            f"the {self.model_name} converter names host {global_name} {names_outside_hf_namespace}, which the HF model of "
            f"{self.args.hf_checkpoint} does not have, so no --target-modules selector can reach its adapter"
        )
        # shared-outer expert adapters are expert-agnostic: one name for the whole grouped host
        module_names = [
            _EXPERT_INDEX_PATTERN.sub(".experts.", name.removesuffix(".weight")) for name in hf_weight_names
        ]
        block_prefixes = {name.rpartition(".")[0] for name in module_names}
        assert len(block_prefixes) == 1 and [name.rpartition(".")[2] for name in module_names] == list(leaves), (
            f"the {self.model_name} converter names host {global_name} {module_names}, but the native LoRA spec "
            f"declares projections {list(leaves)} for it"
        )
        return f"{block_prefixes.pop()}."

    def _global_shape(self, weight: nn.Parameter) -> list[int]:
        shape = list(weight.shape)
        # MCore marks only expert params allreduce=False; they shard over expert TP, pinned to 1 here
        if weight.tensor_model_parallel and getattr(weight, "allreduce", True):
            shape[weight.partition_dim] *= self.tp_size
        return shape


class DeclaredNaming:
    """Adapters named by a fixed table, for architectures whose served adapter names are not
    the HF checkpoint names the converter produces (Inkling: SGLang's module names).

    ``hf_blocks_by_param_prefix`` maps a host's parameter name inside its layer, by prefix, to its HF block.
    """

    def __init__(self, layer_prefix: str, hf_blocks_by_param_prefix: Mapping[str, str]):
        self.layer_prefix = layer_prefix
        self.hf_blocks_by_param_prefix = hf_blocks_by_param_prefix

    def block_prefix(self, layer: nn.Module, host_weight: nn.Parameter, leaves: Sequence[str]) -> str:
        del leaves
        name_in_layer = next(name for name, param in layer.named_parameters() if param is host_weight)
        hf_block = next(
            hf_block
            for param_prefix, hf_block in sorted(
                self.hf_blocks_by_param_prefix.items(), key=lambda item: -len(item[0])
            )
            if name_in_layer.startswith(param_prefix)
        )
        return f"{self.layer_prefix}{layer.layer_number - 1}.{hf_block}"
